//! Reuse the analytic shadow coverage in local physical pixels. Color, position,
//! clipping and compositing remain in the ordinary UI draw, outside the key.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec2, Vec4};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    GpuResourceAppExt, RenderApp,
};

use super::{BoxShadowPipeline, BoxShadowPipelineKey, BoxShadowPipelineMode, ExtractedBoxShadow};
use crate::pipeline::UiWriterEncodeKey;

/// Optional tuning for the automatic soft box-shadow coverage cache.
///
/// The same analytic shader and camera sample count are used to bake each mask.
/// Masks are sampled at local physical-pixel resolution with linear filtering.
/// Hard shadows, sample counts below four and shapes exceeding the budget use
/// the analytic path directly.
#[derive(Resource, Clone, Copy, Debug, Reflect, ExtractResource)]
#[reflect(Resource, Default)]
#[extract_app(RenderApp)]
pub struct BoxShadowCacheSettings {
    pub enabled: bool,
    /// Maximum bytes of mask textures. Unused entries expire after three frames.
    pub max_bytes: usize,
}

impl Default for BoxShadowCacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_bytes: 32 * 1024 * 1024,
        }
    }
}

pub(super) fn install(app: &mut App) {
    app.init_resource::<BoxShadowCacheSettings>()
        .register_type::<BoxShadowCacheSettings>()
        .add_plugins(ExtractResourcePlugin::<BoxShadowCacheSettings>::default());
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .init_gpu_resource::<BoxShadowCache>()
            .add_systems(
                RenderGraph,
                render_masks
                    .after(RenderGraphSystems::Begin)
                    .before(RenderGraphSystems::Render),
            );
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MaskKey {
    size: [u32; 2],
    bounds: [u32; 2],
    radius: [[u32; 4]; 2],
    blur: u32,
    samples: u32,
}

impl MaskKey {
    fn new(shadow: &ExtractedBoxShadow, samples: u32) -> Self {
        let radius: [[f32; 4]; 2] = shadow.radius.into();
        Self {
            size: shadow.size.to_array().map(f32::to_bits),
            bounds: shadow.bounds.to_array().map(f32::to_bits),
            radius: radius.map(|r| r.map(f32::to_bits)),
            blur: shadow.blur_radius.to_bits(),
            samples,
        }
    }

    fn uniform(self) -> MaskUniform {
        MaskUniform {
            size: Vec2::from_array(self.size.map(f32::from_bits)),
            bounds: Vec2::from_array(self.bounds.map(f32::from_bits)),
            radius_x: Vec4::from_array(self.radius[0].map(f32::from_bits)),
            radius_y: Vec4::from_array(self.radius[1].map(f32::from_bits)),
            blur: f32::from_bits(self.blur),
        }
    }
}

#[derive(ShaderType)]
struct MaskUniform {
    size: Vec2,
    bounds: Vec2,
    radius_x: Vec4,
    radius_y: Vec4,
    blur: f32,
}

struct MaskEntry {
    // Retain the texture and uniform buffer with the bind groups.
    _texture: Texture,
    view: TextureView,
    _uniform: UniformBuffer<MaskUniform>,
    uniform_bind_group: BindGroup,
    texture_bind_group: BindGroup,
    pipeline: CachedRenderPipelineId,
    bytes: usize,
    last_used: u64,
    rendered_pipeline: Option<RenderPipelineId>,
}

#[derive(Resource, Default)]
pub struct BoxShadowCache {
    entries: HashMap<MaskKey, MaskEntry>,
    requests: HashMap<MaskKey, u32>,
    previous_requests: HashSet<MaskKey>,
    pub(super) bindings: HashMap<Entity, BindGroup>,
    frame: u64,
    bytes: usize,
    budget: usize,
}

impl BoxShadowCache {
    pub(super) fn begin_frame(&mut self, settings: &BoxShadowCacheSettings) {
        self.frame += 1;
        self.bindings.clear();
        self.previous_requests.clear();
        self.previous_requests.extend(self.requests.keys().copied());
        self.requests.clear();
        self.entries.retain(|_, entry| {
            let keep = settings.enabled
                && settings.max_bytes == self.budget
                && self.frame - entry.last_used < 3;
            if !keep {
                self.bytes -= entry.bytes;
            }
            keep
        });
        self.budget = settings.max_bytes;
    }

    pub(super) fn request(
        &mut self,
        entity: Entity,
        shadow: &ExtractedBoxShadow,
        samples: u32,
        settings: &BoxShadowCacheSettings,
        pipelines: &PipelineCache,
    ) -> bool {
        // Subpixel hard edges need analytic evaluation at the actual position.
        // Very low quadrature counts can overshoot 1, which R8 cannot preserve.
        if !settings.enabled || shadow.blur_radius < 1.0 || samples < 4 {
            return false;
        }
        let key = MaskKey::new(shadow, samples);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = self.frame;
            if entry.rendered_pipeline.is_some_and(|id| {
                pipelines
                    .get_render_pipeline(entry.pipeline)
                    .is_some_and(|pipeline| pipeline.id() == id)
            }) {
                self.bindings
                    .insert(entity, entry.texture_bind_group.clone());
                return true;
            }
        }
        *self.requests.entry(key).or_default() += 1;
        false
    }
}

pub(super) fn prepare_masks(
    mut cache: ResMut<BoxShadowCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<BoxShadowPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BoxShadowPipeline>>,
) {
    // A one-off animated shape stays analytic. Bake repeated shapes immediately,
    // or a single shape after seeing it in two successive frames.
    let cache = &mut *cache;
    for (&key, &count) in &cache.requests {
        if cache.entries.contains_key(&key)
            || (count == 1 && !cache.previous_requests.contains(&key))
        {
            continue;
        }
        let data = key.uniform();
        let size = data.bounds.ceil().as_uvec2();
        let bytes = size.x as usize * size.y as usize;
        if size == UVec2::ZERO
            || size.max_element() > device.limits().max_texture_dimension_2d
            || cache.bytes + bytes > cache.budget
        {
            continue;
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("ui_box_shadow_mask"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let mut uniform = UniformBuffer::from(data);
        uniform.write_buffer(&device, &queue);
        let uniform_bind_group = device.create_bind_group(
            "ui_box_shadow_mask_uniform",
            &pipeline_cache.get_bind_group_layout(&pipeline.mask_layout),
            &BindGroupEntries::single(uniform.binding().unwrap()),
        );
        let texture_bind_group = device.create_bind_group(
            "ui_box_shadow_mask_texture",
            &pipeline_cache.get_bind_group_layout(&pipeline.texture_layout),
            &BindGroupEntries::sequential((&view, &pipeline.mask_sampler)),
        );
        let mask_pipeline = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            BoxShadowPipelineKey {
                target_format: TextureFormat::R8Unorm,
                samples: key.samples,
                writer_encode: UiWriterEncodeKey::default(),
                mode: BoxShadowPipelineMode::Mask,
            },
        );
        let frame = cache.frame;
        cache.entries.insert(
            key,
            MaskEntry {
                _texture: texture,
                view,
                _uniform: uniform,
                uniform_bind_group,
                texture_bind_group,
                pipeline: mask_pipeline,
                bytes,
                last_used: frame,
                rendered_pipeline: None,
            },
        );
        cache.bytes += bytes;
    }
}

fn render_masks(
    mut cache: ResMut<BoxShadowCache>,
    pipelines: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    for entry in cache.entries.values_mut() {
        let Some(pipeline) = pipelines.get_render_pipeline(entry.pipeline) else {
            continue;
        };
        if entry.rendered_pipeline == Some(pipeline.id()) {
            continue;
        }
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ui_box_shadow_mask"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &entry.view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &entry.uniform_bind_group, &[]);
        pass.draw(0..3, 0..1);
        entry.rendered_pipeline = Some(pipeline.id());
    }
}

pub(super) fn mask_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ui_box_shadow_mask_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            uniform_buffer::<MaskUniform>(false),
        ),
    )
}

pub(super) fn texture_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ui_box_shadow_texture_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}
