use core::ops::Range;
use std::borrow::Cow;

use super::{ImageNodeBindGroups, UiBatch, UiMeta, UiViewTarget};

use crate::UiCameraView;
use bevy_ecs::{
    entity::EntityHash,
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::FloatOrd;
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::*,
    render_resource::{CachedRenderPipelineId, RenderPassDescriptor},
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::*,
};
use indexmap::IndexMap;
use tracing::error;

/// GPU work executed between two UI render passes.
///
/// The entity is the UI view entity, whose [`UiViewTarget`] identifies the target
/// camera. The callback can read prepared render-world data and record commands
/// through the same [`RenderContext`] as the UI pass. No render pass is open
/// during the callback. Resources used by a callback must be prepared before
/// rendering; a callback cannot mutate ECS state.
pub type UiPassCallback = fn(&World, Entity, &mut RenderContext<'_, '_>);

struct UiPassSegment {
    before_index: usize,
    callback: Option<UiPassCallback>,
}

/// Per-frame UI batch boundaries and optional work between render passes.
///
/// Register boundaries after [`RenderSystems::PhaseSort`](bevy_render::RenderSystems::PhaseSort)
/// and before [`RenderSystems::Prepare`](bevy_render::RenderSystems::Prepare).
/// Indices refer to the sorted [`TransparentUi`] items, before batching. All
/// native UI batchers preserve these boundaries. Requests are cleared during
/// [`RenderSystems::Cleanup`](bevy_render::RenderSystems::Cleanup), so extensions
/// must register the requests they need each frame.
///
/// This is a render-world resource; applications normally use it through a
/// rendering plugin rather than changing their main-world UI nodes.
#[derive(Resource, Default)]
pub struct UiPassSegments {
    views: HashMap<RetainedViewEntity, Vec<UiPassSegment>>,
}

impl UiPassSegments {
    /// Prevents batching across `before_index` and optionally runs GPU work there.
    ///
    /// `before_index` must be in `0..=phase.items.len()`. A callback runs after all
    /// earlier items have been drawn and before the item at this index. Requests
    /// at the same index execute in registration order. With `None`, only batches
    /// are split; rendering can continue in a single render pass.
    pub fn add(
        &mut self,
        view: RetainedViewEntity,
        before_index: usize,
        callback: Option<UiPassCallback>,
    ) {
        let segments = self.views.entry(view).or_default();
        let insertion = segments.partition_point(|segment| segment.before_index <= before_index);
        segments.insert(
            insertion,
            UiPassSegment {
                before_index,
                callback,
            },
        );
    }

    pub(crate) fn breaks_batch(&self, view: RetainedViewEntity, before_index: usize) -> bool {
        self.views.get(&view).is_some_and(|segments| {
            segments
                .binary_search_by_key(&before_index, |segment| segment.before_index)
                .is_ok()
        })
    }
}

pub(crate) fn clear_ui_pass_segments(mut segments: ResMut<UiPassSegments>) {
    segments.views.clear();
}

pub fn ui_pass(
    world: &World,
    view: ViewQuery<&UiCameraView>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    ui_view_target_query: Query<(&ViewTarget, &ExtractedCamera)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    segments: Res<UiPassSegments>,
    mut ctx: RenderContext,
) {
    let ui_camera_view = view.into_inner();
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        return;
    };

    let Ok((target, camera)) = ui_view_target_query.get(ui_view_target.0) else {
        return;
    };

    let Some(transparent_phase) =
        transparent_render_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };

    let callbacks = segments
        .views
        .get(&extracted_view.retained_view_entity)
        .into_iter()
        .flatten()
        .filter_map(|segment| {
            segment
                .callback
                .map(|callback| (segment.before_index, callback))
        });
    let mut callbacks = callbacks.peekable();

    // Keep the existing single-pass path for views without GPU callbacks,
    // including those that only request batch boundaries.
    if callbacks.peek().is_none() {
        if transparent_phase.items.is_empty() {
            return;
        }
        draw_ui_range(
            world,
            &mut ctx,
            transparent_phase,
            ui_view_entity,
            target,
            camera,
            0..transparent_phase.items.len(),
            "ui".into(),
        );
        return;
    }

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "ui");
    let mut start = 0;
    let mut segment_index = 0;
    for (before_index, callback) in callbacks {
        if start < before_index {
            draw_ui_range(
                world,
                &mut ctx,
                transparent_phase,
                ui_view_entity,
                target,
                camera,
                start..before_index,
                format!("ui_segment_{segment_index}").into(),
            );
            segment_index += 1;
        }
        callback(world, ui_view_entity, &mut ctx);
        start = before_index;
    }
    if start < transparent_phase.items.len() {
        draw_ui_range(
            world,
            &mut ctx,
            transparent_phase,
            ui_view_entity,
            target,
            camera,
            start..transparent_phase.items.len(),
            format!("ui_segment_{segment_index}").into(),
        );
    }
    span.end(ctx.command_encoder());
}

fn draw_ui_range(
    world: &World,
    ctx: &mut RenderContext,
    phase: &SortedRenderPhase<TransparentUi>,
    ui_view_entity: Entity,
    target: &ViewTarget,
    camera: &ExtractedCamera,
    range: Range<usize>,
    name: Cow<'static, str>,
) {
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(target.get_unsampled_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, name);

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = phase.render_range(&mut render_pass, world, ui_view_entity, range) {
        error!("Error encountered while rendering the ui phase {err:?}");
    }

    pass_span.end(&mut render_pass);
}

#[derive(Debug)]
pub struct TransparentUi {
    pub sort_key: FloatOrd,
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub indexed: bool,
}

impl PhaseItem for TransparentUi {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TransparentUi {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut IndexMap<(Entity, MainEntity), TransparentUi, EntityHash>) {
        items.sort_by_key(|_, value| value.sort_key());
    }

    fn recalculate_sort_keys(
        _: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _: &ExtractedView,
    ) {
        // Sort keys are precalculated for UI phase items.
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub type DrawUi = (
    SetItemPipeline,
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<1>,
    DrawUiNode,
);

pub struct SetUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<I> {
    type Param = SRes<UiMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
pub struct SetUiTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<I> {
    type Param = SRes<ImageNodeBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, image_bind_groups.values.get(&batch.image).unwrap(), &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawUiNode;
impl<P: PhaseItem> RenderCommand<P> for DrawUiNode {
    type Param = SRes<UiMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));
        // Define how to "connect" the vertices
        pass.set_index_buffer(
            indices.slice(..),
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
