use super::*;
use bevy_ui::CalculatedClipRect;

fn glyph(translation: Vec2, atlas_rect: Rect) -> ExtractedGlyph {
    ExtractedGlyph {
        color: LinearRgba::WHITE,
        translation,
        rect: atlas_rect,
    }
}

fn node(
    glyphs: Vec<ExtractedGlyph>,
    transform: Affine2,
    clip: Option<CalculatedClip>,
) -> ExtractedUiNode {
    ExtractedUiNode {
        z_order: 1.,
        image: AssetId::default(),
        clip,
        item: ExtractedUiItem::glyphs(glyphs),
        transform,
    }
}

fn clip(rect: Rect, world_to_clip_local: Affine2) -> CalculatedClip {
    CalculatedClip::Rects(
        vec![CalculatedClipRect {
            rect,
            world_to_clip_local,
        }]
        .into(),
    )
}

#[test]
fn glyph_bounds_use_rendered_positions_including_overhangs_not_atlas_offsets() {
    let item = ExtractedUiItem::glyphs(vec![
        glyph(Vec2::new(-8., 3.), Rect::new(100., 200., 106., 210.)),
        glyph(Vec2::new(12., -4.), Rect::new(400., 20., 410., 24.)),
    ]);
    let ExtractedUiItem::Glyphs { bounds, .. } = item else {
        unreachable!()
    };
    assert_eq!(bounds, Rect::new(-11., -6., 17., 8.));
}

#[test]
fn whole_run_is_rejected_outside_and_returns_when_scrolled_into_view() {
    let mut run = node(
        vec![glyph(Vec2::ZERO, Rect::new(100., 100., 110., 110.))],
        Affine2::from_translation(Vec2::new(0., 30.)),
        Some(clip(Rect::new(-10., -10., 10., 10.), Affine2::IDENTITY)),
    );
    assert!(run.glyph_run_is_fully_clipped());
    run.transform = Affine2::IDENTITY;
    assert!(!run.glyph_run_is_fully_clipped());
    run.clip = Some(CalculatedClip::FullyClipped);
    assert!(run.glyph_run_is_fully_clipped());
    run.clip = None;
    assert!(!run.glyph_run_is_fully_clipped());
}

#[test]
fn transformed_partial_and_touching_runs_keep_the_original_polygon_path() {
    let glyphs = || vec![glyph(Vec2::ZERO, Rect::new(0., 0., 20., 12.))];
    let region = clip(Rect::new(-10., -10., 10., 10.), Affine2::from_angle(0.6));
    for position in [Vec2::ZERO, Vec2::new(10., 0.), Vec2::new(20., 0.)] {
        let run = node(
            glyphs(),
            Affine2::from_scale_angle_translation(Vec2::new(1.5, 0.7), -0.3, position),
            Some(region.clone()),
        );
        let ExtractedUiItem::Glyphs { glyphs, .. } = &run.item else {
            unreachable!()
        };
        let points = QUAD_VERTEX_POSITIONS.map(|p| {
            (
                run.transform.transform_point2(p * glyphs[0].rect.size()),
                Vec2::ZERO,
            )
        });
        let old = clip_polygon(run.clip.as_ref(), &points, Vec2::lerp);
        assert!(
            !run.glyph_run_is_fully_clipped() || old.is_empty(),
            "visible geometry was rejected"
        );
    }
    let touching = node(
        glyphs(),
        Affine2::from_translation(Vec2::new(20., 0.)),
        Some(clip(Rect::new(-10., -10., 10., 10.), Affine2::IDENTITY)),
    );
    assert!(!touching.glyph_run_is_fully_clipped());
}

#[test]
fn a_visible_overflowing_glyph_keeps_the_entire_run_eligible() {
    let run = node(
        vec![
            glyph(Vec2::ZERO, Rect::new(20., 20., 30., 30.)),
            glyph(Vec2::new(100., 0.), Rect::new(100., 100., 120., 120.)),
        ],
        Affine2::IDENTITY,
        Some(clip(Rect::new(95., -5., 105., 5.), Affine2::IDENTITY)),
    );
    assert!(!run.glyph_run_is_fully_clipped());
}

#[test]
fn bounds_classification_preserves_polygon_output_under_scroll_rotation_and_scale() {
    use clipping::{classify_bounds, BoundsClip};
    let clip = CalculatedClip::Rects(
        vec![
            CalculatedClipRect {
                rect: Rect::new(-30., -20., 30., 20.),
                world_to_clip_local: Affine2::IDENTITY,
            },
            CalculatedClipRect {
                rect: Rect::new(-28., -24., 28., 24.),
                world_to_clip_local: Affine2::from_angle(0.2),
            },
        ]
        .into(),
    );
    let mut inside = 0;
    let mut outside = 0;
    let mut partial = 0;
    for angle in [-0.7, 0., 0.4] {
        for offset in -12..=12 {
            let run = node(
                vec![
                    glyph(Vec2::new(-8., 0.), Rect::new(100., 100., 110., 114.)),
                    glyph(Vec2::new(8., 2.), Rect::new(200., 40., 212., 52.)),
                ],
                Affine2::from_scale_angle_translation(
                    Vec2::new(1.3, 0.8),
                    angle,
                    Vec2::new(offset as f32 * 5., 0.),
                ),
                Some(clip.clone()),
            );
            let ExtractedUiItem::Glyphs { bounds, glyphs } = &run.item else {
                unreachable!()
            };
            let classification = classify_bounds(run.clip.as_ref(), run.transform, *bounds);
            match classification {
                BoundsClip::Inside => inside += 1,
                BoundsClip::Outside => outside += 1,
                BoundsClip::Partial => partial += 1,
            }
            for glyph in glyphs {
                let points = QUAD_VERTEX_POSITIONS.map(|p| {
                    (
                        run.transform
                            .transform_point2(glyph.translation + p * glyph.rect.size()),
                        p,
                    )
                });
                let original = clip_polygon(run.clip.as_ref(), &points, Vec2::lerp);
                match classification {
                    BoundsClip::Outside => {
                        assert!(original.is_empty(), "visible glyph was rejected")
                    }
                    BoundsClip::Inside => {
                        assert_eq!(original.as_slice(), &points, "clipping was needed")
                    }
                    BoundsClip::Partial => {}
                }
            }
        }
    }
    assert!(inside > 0 && outside > 0 && partial > 0);
}
