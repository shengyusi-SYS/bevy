# Optional pointer click filter

This branch starts at `19f048151dc9037c728e5ac54456b42785bdb60c` and keeps Bevy 0.20.1.

`bevy_picking::events::PointerClickFilter` is an optional resource containing a short
read-only `fn(&World, &PointerClick) -> bool` predicate. It runs immediately before
the queued click dispatch, after preceding pointer observers have resolved.

- Without the resource, the original click trigger and immediate message path remain unchanged.
- With it, an accepted click writes its original message and then triggers observers;
  a rejected click reaches neither channel. Click messages become visible at this
  dispatch sync point rather than during `pointer_events` itself.
- Release, drag/drop, cancellation, hit testing, and click counts retain their behavior.
- Applications own UI hit-owner, movement-tolerance, and gesture policies. The hook
  does not implement those policies or change keyboard/accessibility activation.

The regression exercises default, rejected, and observer-dependent accepted clicks,
including preserved release/drag-end and message visibility during click observers.

Related upstream discussion: <https://github.com/bevyengine/bevy/pull/21149>.
That proposal adds drag metadata; it does not itself provide an equivalent pre-widget
filter. Remove this patch when upstream exposes equivalent eligibility integration
and the same dispatch/lifecycle regressions pass against it.
