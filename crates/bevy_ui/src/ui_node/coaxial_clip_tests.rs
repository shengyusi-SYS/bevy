use super::{CalculatedClip, CalculatedClipRect};
use bevy_math::{Affine2, Mat2, Rect, Vec2};

fn raw(regions: &[CalculatedClipRect]) -> CalculatedClip {
    CalculatedClip::Rects(regions.iter().copied().collect())
}

#[test]
fn translated_regions_in_one_basis_compact_to_their_intersection() {
    let basis = Mat2::from_diagonal(Vec2::new(0.5, 2.));
    let first = Affine2::from_mat2_translation(basis, Vec2::new(2., -4.));
    let second = Affine2::from_mat2_translation(basis, Vec2::new(7., -1.));
    let mut clip = CalculatedClip::default();
    clip.push_rect(Rect::new(-10., -10., 10., 10.), first);
    clip.push_rect(Rect::new(-8., -8., 8., 8.), second);
    assert_eq!(clip.rects().unwrap().len(), 1);
    assert_eq!(clip.rects().unwrap()[0].rect, Rect::new(-10., -10., 3., 5.));
    assert_eq!(clip.rects().unwrap()[0].world_to_clip_local, first);
}

#[test]
fn differing_bases_keep_their_individual_regions() {
    let regions = [
        CalculatedClipRect {
            rect: Rect::new(-20., -20., 20., 20.),
            world_to_clip_local: Affine2::IDENTITY,
        },
        CalculatedClipRect {
            rect: Rect::new(-10., -10., 10., 10.),
            world_to_clip_local: Affine2::from_angle(0.4),
        },
    ];
    let mut clip = CalculatedClip::default();
    for r in regions {
        clip.push_rect(r.rect, r.world_to_clip_local);
    }
    assert_eq!(clip, raw(&regions));
}

#[test]
fn disjoint_same_basis_regions_stop_descendant_clipping_work() {
    let mut clip = CalculatedClip::default();
    clip.push_rect(Rect::new(-10., -10., 10., 10.), Affine2::IDENTITY);
    clip.push_rect(Rect::new(30., -5., 40., 5.), Affine2::IDENTITY);
    assert_eq!(clip, CalculatedClip::FullyClipped);
    clip.push_rect(Rect::new(-1., -1., 1., 1.), Affine2::IDENTITY);
    assert_eq!(clip, CalculatedClip::FullyClipped);
}

#[test]
fn perpendicular_unbounded_axes_compact_without_infinite_coordinate_transforms() {
    let mut clip = CalculatedClip::default();
    clip.push_rect(
        Rect::new(-10., f32::NEG_INFINITY, 10., f32::INFINITY),
        Affine2::IDENTITY,
    );
    clip.push_rect(
        Rect::new(f32::NEG_INFINITY, -5., f32::INFINITY, 5.),
        Affine2::IDENTITY,
    );
    assert_eq!(clip.rects().unwrap().len(), 1);
    assert_eq!(clip.rects().unwrap()[0].rect, Rect::new(-10., -5., 10., 5.));
}

#[test]
fn compaction_preserves_point_membership_for_shared_rotated_scaled_basis() {
    let basis = Mat2::from_angle(0.37) * Mat2::from_diagonal(Vec2::new(0.5, 1.5));
    let regions = [
        CalculatedClipRect {
            rect: Rect::new(-30., -25., 30., 25.),
            world_to_clip_local: Affine2::from_mat2_translation(basis, Vec2::new(-1000., 800.)),
        },
        CalculatedClipRect {
            rect: Rect::new(-25., -20., 25., 20.),
            world_to_clip_local: Affine2::from_mat2_translation(basis, Vec2::new(-1004., 803.)),
        },
        CalculatedClipRect {
            rect: Rect::new(-22., -16., 22., 16.),
            world_to_clip_local: Affine2::from_mat2_translation(basis, Vec2::new(-998., 799.)),
        },
    ];
    let original = raw(&regions);
    let mut compacted = CalculatedClip::default();
    for r in regions {
        compacted.push_rect(r.rect, r.world_to_clip_local);
    }
    assert_eq!(compacted.rects().unwrap().len(), 1);
    let origin = regions[0]
        .world_to_clip_local
        .inverse()
        .transform_point2(Vec2::ZERO);
    let mut visible = 0;
    for x in -8..=8 {
        for y in -8..=8 {
            let point = origin + Vec2::new(x as f32 * 4.1, y as f32 * 3.7);
            let expected = original.contains_point(point);
            visible += usize::from(expected);
            assert_eq!(compacted.contains_point(point), expected);
        }
    }
    assert!(visible > 0 && visible < 17 * 17);
}
