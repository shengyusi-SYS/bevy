//! Extraction adapter contract, without a renderer, window, fonts or GPU.
use super::*;
use bevy_extract::MainWorld;

fn step(render: &mut World) {
    render.resource_mut::<MainWorld>().increment_change_tick();
    render.run_system_cached(extract_uinode_changes).unwrap();
    render.resource_mut::<MainWorld>().clear_trackers();
}

fn fixture() -> (World, Entity, Entity) {
    let mut main = MainWorld::default();
    #[cfg(feature = "bevy_ui_debug")]
    main.init_resource::<GlobalUiDebugOptions>();
    let source = main
        .spawn((
            Node::default(),
            ComputedNode {
                size: Vec2::new(100., 40.),
                unrounded_size: Vec2::new(100., 40.),
                ..Default::default()
            },
            InheritedVisibility::VISIBLE,
            UiGlobalTransform::from(Affine2::from_translation(Vec2::new(10., 20.))),
            BackgroundColor(bevy_color::Color::WHITE),
        ))
        .id();
    let mut render = World::new();
    render.insert_resource(main);
    render.init_resource::<ExtractedUiNodes>();
    step(&mut render);
    let extracted = render.spawn_empty().id();
    let mut nodes = EntityIndexMap::default();
    nodes.insert(
        extracted,
        ExtractedUiNode {
            z_order: 3.06,
            image: AssetId::default(),
            clip: None,
            transform: Affine2::from_scale_angle_translation(
                Vec2::splat(2.),
                0.,
                Vec2::new(12., 24.),
            ),
            item: ExtractedUiItem::glyphs(vec![ExtractedGlyph {
                color: LinearRgba::RED,
                translation: Vec2::new(2., 4.),
                rect: Rect::new(1., 2., 5., 8.),
            }]),
        },
    );
    render
        .resource_mut::<ExtractedUiNodes>()
        .uinodes
        .insert(source.into(), (Entity::PLACEHOLDER, nodes));
    (render, source, extracted)
}

#[test]
fn scrolling_retains_glyph_entities_and_updates_geometry_from_original_positions() {
    let (mut render, source, extracted) = fixture();
    for (source_position, expected_position) in [
        (Vec2::new(90., 30.), Vec2::new(92., 34.)),
        (Vec2::new(-10., 40.), Vec2::new(-8., 44.)),
        (Vec2::new(10., 20.), Vec2::new(12., 24.)),
    ] {
        render
            .resource_mut::<MainWorld>()
            .entity_mut(source)
            .insert((
                UiGlobalTransform::from(Affine2::from_translation(source_position)),
                CalculatedClip::FullyClipped,
            ));
        step(&mut render);
        let retained = render.resource::<ExtractedUiNodes>();
        let nodes = &retained
            .uinodes
            .get(&MainEntity::from(source))
            .expect("scrolling must retain extracted content")
            .1;
        assert!(render.get_entity(extracted).is_ok());
        assert!(
            !retained.changed.contains(&MainEntity::from(source)),
            "geometry-only updates must not repeat content extraction"
        );
        assert_eq!(nodes.len(), 1);
        let node = &nodes[&extracted];
        assert_eq!(node.transform.translation, expected_position);
        assert_eq!(
            node.transform.matrix2,
            bevy_math::Mat2::from_diagonal(Vec2::splat(2.))
        );
        assert_eq!(node.clip, Some(CalculatedClip::FullyClipped));
        assert_eq!(node.z_order, 3.06);
        let ExtractedUiItem::Glyphs { glyphs, .. } = &node.item else {
            panic!("glyph content was replaced");
        };
        assert_eq!(glyphs[0].color, LinearRgba::RED);
        assert_eq!(glyphs[0].translation, Vec2::new(2., 4.));
        assert_eq!(glyphs[0].rect, Rect::new(1., 2., 5., 8.));
    }
}

#[test]
fn content_changes_keep_using_full_extraction() {
    let (mut render, source, extracted) = fixture();
    render
        .resource_mut::<MainWorld>()
        .entity_mut(source)
        .insert((
            UiGlobalTransform::from(Affine2::from_translation(Vec2::new(90., 30.))),
            BackgroundColor(bevy_color::Color::BLACK),
        ));
    step(&mut render);
    let retained = render.resource::<ExtractedUiNodes>();
    assert!(retained.changed.contains(&MainEntity::from(source)));
    assert!(!retained.uinodes.contains_key(&MainEntity::from(source)));
    assert!(render.get_entity(extracted).is_err());
}

#[test]
fn linear_transform_changes_keep_using_full_extraction() {
    let (mut render, source, extracted) = fixture();
    render
        .resource_mut::<MainWorld>()
        .entity_mut(source)
        .insert(UiGlobalTransform::from(Affine2::from_scale(Vec2::splat(
            0.5,
        ))));
    step(&mut render);
    let retained = render.resource::<ExtractedUiNodes>();
    assert!(retained.changed.contains(&MainEntity::from(source)));
    assert!(!retained.uinodes.contains_key(&MainEntity::from(source)));
    assert!(render.get_entity(extracted).is_err());
}
