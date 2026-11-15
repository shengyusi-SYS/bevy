use alloc::{boxed::Box, vec::Vec};
use bevy_platform::cell::SyncUnsafeCell;
use bevy_platform::sync::Arc;
use bevy_tasks::{ComputeTaskPool, Scope, TaskPool, ThreadExecutor};
use concurrent_queue::ConcurrentQueue;
use core::{any::Any, panic::AssertUnwindSafe};
use fixedbitset::FixedBitSet;
use std::sync::{Mutex, MutexGuard};

#[cfg(feature = "trace")]
use tracing::{info_span, Span};

use crate::{
    error::{BevyError, ErrorContext, ErrorHandler, Result},
    prelude::Resource,
    schedule::{
        is_apply_deferred, ConditionWithAccess, SystemExecutor, SystemSchedule, SystemWithAccess,
    },
    system::{BoxedSystem, RunSystemError, ScheduleSystem},
    world::{unsafe_world_cell::UnsafeWorldCell, World},
};
#[cfg(feature = "hotpatching")]
use crate::{prelude::DetectChanges, HotPatchChanges};

use super::__rust_begin_short_backtrace;

const MAX_INLINE_SYSTEMS: usize = 16;

/// Borrowed data used by the [`MultiThreadedExecutor`].
struct Environment<'env, 'sys> {
    executor: &'env MultiThreadedExecutor,
    systems: &'sys [SyncUnsafeCell<SystemWithAccess>],
    conditions: SyncUnsafeCell<Conditions<'sys>>,
    world_cell: UnsafeWorldCell<'env>,
}

struct Conditions<'a> {
    system_conditions: &'a mut [Vec<ConditionWithAccess>],
    set_conditions: &'a mut [Vec<ConditionWithAccess>],
    sets_with_conditions_of_systems: &'a [FixedBitSet],
    systems_in_sets_with_conditions: &'a [FixedBitSet],
}

impl<'env, 'sys> Environment<'env, 'sys> {
    fn new(
        executor: &'env MultiThreadedExecutor,
        schedule: &'sys mut SystemSchedule,
        world: &'env mut World,
    ) -> Self {
        Environment {
            executor,
            systems: SyncUnsafeCell::from_mut(schedule.systems.as_mut_slice()).as_slice_of_cells(),
            conditions: SyncUnsafeCell::new(Conditions {
                system_conditions: &mut schedule.system_conditions,
                set_conditions: &mut schedule.set_conditions,
                sets_with_conditions_of_systems: &schedule.sets_with_conditions_of_systems,
                systems_in_sets_with_conditions: &schedule.systems_in_sets_with_conditions,
            }),
            world_cell: world.as_unsafe_world_cell(),
        }
    }
}

/// Per-system data used by the [`MultiThreadedExecutor`].
// Copied here because it can't be read from the system when it's running.
struct SystemTaskMetadata {
    /// The set of systems whose `component_access_set()` conflicts with this one.
    conflicting_systems: FixedBitSet,
    /// The set of systems whose `component_access_set()` conflicts with this system's conditions.
    /// Note that this is separate from `conflicting_systems` to handle the case where
    /// a system is skipped by an earlier system set condition or system stepping,
    /// and needs access to run its conditions but not for itself.
    condition_conflicting_systems: FixedBitSet,
    /// Indices of the systems that directly depend on the system.
    dependents: Vec<usize>,
    /// Is `true` if the system does not access `!Send` data.
    is_send: bool,
    /// Is `true` if the system is exclusive.
    is_exclusive: bool,
}

/// The result of running a system that is sent across a channel.
struct SystemResult {
    system_index: usize,
}

/// Runs the schedule using a thread pool. Non-conflicting systems can run in parallel.
pub struct MultiThreadedExecutor {
    /// The running state, protected by a mutex so that a reference to the executor can be shared across tasks.
    state: Mutex<ExecutorState>,
    /// Queue of system completion events.
    system_completion: ConcurrentQueue<SystemResult>,
    /// Setting when true applies deferred system buffers after all systems have run
    apply_final_deferred: bool,
    /// Allow the scope caller to execute a Send system when starting an idle frontier.
    scope_thread_execution: bool,
    inline_system_continuations: bool,
    #[cfg(test)]
    task_spawns: Option<Arc<core::sync::atomic::AtomicUsize>>,
    #[cfg(test)]
    peak_pending_send_tasks: Option<Arc<core::sync::atomic::AtomicUsize>>,
    /// When set, tells the executor that a thread has panicked.
    panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
    starting_systems: FixedBitSet,
    /// Cached tracing span
    #[cfg(feature = "trace")]
    executor_span: Span,
}

/// The state of the executor while running.
pub struct ExecutorState {
    /// Metadata for scheduling and running system tasks.
    system_task_metadata: Vec<SystemTaskMetadata>,
    /// The set of systems whose `component_access_set()` conflicts with this system set's conditions.
    set_condition_conflicting_systems: Vec<FixedBitSet>,
    /// Returns `true` if a system with non-`Send` access is running.
    local_thread_running: bool,
    /// Returns `true` if an exclusive system is running.
    exclusive_running: bool,
    /// The number of systems that are running.
    num_running_systems: usize,
    /// Running Send systems, excluding thread-affine/exclusive work.
    num_running_send_systems: usize,
    /// Keep ready roots available for continuation rather than queueing all of them.
    max_running_send_systems: usize,
    /// The number of dependencies each system has that have not been satisfied. A dependency
    /// is satisfied when the predecessor completes.
    num_dependencies_remaining: Vec<usize>,
    /// System sets whose conditions have been evaluated.
    evaluated_sets: FixedBitSet,
    /// Systems that have no remaining dependencies and are waiting to run.
    ready_systems: FixedBitSet,
    /// copy of `ready_systems`
    ready_systems_copy: FixedBitSet,
    /// Systems that are running.
    running_systems: FixedBitSet,
    /// Systems that got skipped.
    skipped_systems: FixedBitSet,
    /// Systems whose conditions have been evaluated and were run or skipped.
    completed_systems: FixedBitSet,
    /// Systems that have run but have not had their buffers applied.
    unapplied_systems: FixedBitSet,
}

/// References to data required by the executor.
/// This is copied to each system task so that can invoke the executor when they complete.
// These all need to outlive 'scope in order to be sent to new tasks,
// and keeping them all in a struct means we can use lifetime elision.
#[derive(Copy, Clone)]
struct Context<'scope, 'env, 'sys> {
    environment: &'env Environment<'env, 'sys>,
    scope: &'scope Scope<'scope, 'env, ()>,
    error_handler: ErrorHandler,
}

impl Default for MultiThreadedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemExecutor for MultiThreadedExecutor {
    fn init(&mut self, schedule: &SystemSchedule) {
        let state = self.state.get_mut().unwrap();
        // pre-allocate space
        let sys_count = schedule.system_ids.len();
        let set_count = schedule.set_ids.len();

        self.system_completion = ConcurrentQueue::bounded(sys_count.max(1));
        self.starting_systems = FixedBitSet::with_capacity(sys_count);
        state.evaluated_sets = FixedBitSet::with_capacity(set_count);
        state.ready_systems = FixedBitSet::with_capacity(sys_count);
        state.ready_systems_copy = FixedBitSet::with_capacity(sys_count);
        state.running_systems = FixedBitSet::with_capacity(sys_count);
        state.completed_systems = FixedBitSet::with_capacity(sys_count);
        state.skipped_systems = FixedBitSet::with_capacity(sys_count);
        state.unapplied_systems = FixedBitSet::with_capacity(sys_count);

        state.system_task_metadata = Vec::with_capacity(sys_count);
        for index in 0..sys_count {
            state.system_task_metadata.push(SystemTaskMetadata {
                conflicting_systems: FixedBitSet::with_capacity(sys_count),
                condition_conflicting_systems: FixedBitSet::with_capacity(sys_count),
                dependents: schedule.system_dependents[index].clone(),
                is_send: schedule.systems[index].system.is_send(),
                is_exclusive: schedule.systems[index].access.is_exclusive(),
            });
            // A system with no dependencies is a starting system.
            if schedule.system_dependencies[index] == 0 {
                self.starting_systems.insert(index);
            }
        }

        {
            #[cfg(feature = "trace")]
            let _span = info_span!("calculate conflicting systems").entered();
            for index1 in 0..sys_count {
                let system1 = &schedule.systems[index1];
                for index2 in 0..index1 {
                    let system2 = &schedule.systems[index2];
                    if !system2.access.is_compatible(&system1.access) {
                        state.system_task_metadata[index1]
                            .conflicting_systems
                            .insert(index2);
                        state.system_task_metadata[index2]
                            .conflicting_systems
                            .insert(index1);
                    }
                }

                for index2 in 0..sys_count {
                    let system2 = &schedule.systems[index2];
                    if schedule.system_conditions[index1]
                        .iter()
                        .any(|condition| !system2.access.is_compatible(&condition.access))
                    {
                        state.system_task_metadata[index1]
                            .condition_conflicting_systems
                            .insert(index2);
                    }
                }
            }

            state.set_condition_conflicting_systems.clear();
            state.set_condition_conflicting_systems.reserve(set_count);
            for set_idx in 0..set_count {
                let mut conflicting_systems = FixedBitSet::with_capacity(sys_count);
                for sys_index in 0..sys_count {
                    let system = &schedule.systems[sys_index];
                    if schedule.set_conditions[set_idx]
                        .iter()
                        .any(|condition| !system.access.is_compatible(&condition.access))
                    {
                        conflicting_systems.insert(sys_index);
                    }
                }
                state
                    .set_condition_conflicting_systems
                    .push(conflicting_systems);
            }
        }

        state.num_dependencies_remaining = Vec::with_capacity(sys_count);
    }

    fn run(
        &mut self,
        schedule: &mut SystemSchedule,
        world: &mut World,
        _skip_systems: Option<&FixedBitSet>,
        error_handler: ErrorHandler,
    ) {
        let state = self.state.get_mut().unwrap();
        // reset counts
        if schedule.systems.is_empty() {
            return;
        }
        let task_pool = ComputeTaskPool::get_or_init(TaskPool::default);
        state.num_running_systems = 0;
        state.num_running_send_systems = 0;
        state.max_running_send_systems = if self.inline_system_continuations {
            task_pool
                .thread_num()
                .saturating_add(usize::from(self.scope_thread_execution))
                .max(1)
        } else {
            usize::MAX
        };
        state
            .num_dependencies_remaining
            .clone_from(&schedule.system_dependencies);
        state.ready_systems.clone_from(&self.starting_systems);

        // If stepping is enabled, make sure we skip those systems that should
        // not be run.
        #[cfg(feature = "bevy_debug_stepping")]
        if let Some(skipped_systems) = _skip_systems {
            debug_assert_eq!(skipped_systems.len(), state.completed_systems.len());
            // mark skipped systems as completed
            state.completed_systems |= skipped_systems;

            // signal the dependencies for each of the skipped systems, as
            // though they had run
            for system_index in skipped_systems.ones() {
                state.signal_dependents(system_index);
                state.ready_systems.remove(system_index);
            }
        }

        let thread_executor = world
            .get_resource::<MainThreadExecutor>()
            .map(|e| e.0.clone());
        let thread_executor = thread_executor.as_deref();

        let environment = &Environment::new(self, schedule, world);

        task_pool.scope_with_executor(false, thread_executor, |scope| {
            let context = Context {
                environment,
                scope,
                error_handler,
            };

            // The first tick won't need to process finished systems, but we still need to run the loop in
            // tick_executor() in case a system completes while the first tick still holds the mutex.
            let _ = context.tick_executor(false);
        });

        // End the borrows of self and world in environment by copying out the reference to systems.
        let systems = environment.systems;

        let state = self.state.get_mut().unwrap();
        if self.apply_final_deferred {
            // Do one final apply buffers after all systems have completed
            // Commands should be applied while on the scope's thread, not the executor's thread
            let res = apply_deferred(&state.unapplied_systems, systems, world, error_handler);
            if let Err(payload) = res {
                let panic_payload = self.panic_payload.get_mut().unwrap();
                *panic_payload = Some(payload);
            }
            state.unapplied_systems.clear();
        }

        // check to see if there was a panic
        let payload = self.panic_payload.get_mut().unwrap();
        if let Some(payload) = payload.take() {
            std::panic::resume_unwind(payload);
        }

        debug_assert!(state.ready_systems.is_clear());
        debug_assert!(state.running_systems.is_clear());
        state.evaluated_sets.clear();
        state.skipped_systems.clear();
        state.completed_systems.clear();
    }

    fn set_apply_final_deferred(&mut self, value: bool) {
        self.apply_final_deferred = value;
    }
}

impl<'scope, 'env: 'scope, 'sys> Context<'scope, 'env, 'sys> {
    fn system_completed(
        &self,
        system_index: usize,
        res: Result<(), Box<dyn Any + Send>>,
        allow_inline: bool,
    ) -> Option<usize> {
        // tell the executor that the system finished
        self.environment
            .executor
            .system_completion
            .push(SystemResult { system_index })
            .unwrap_or_else(|error| unreachable!("{}", error));
        if let Err(payload) = res {
            // set the payload to propagate the error
            let mut panic_payload = self.environment.executor.panic_payload.lock().unwrap();
            *panic_payload = Some(payload);
        }
        self.tick_executor(allow_inline)
    }

    #[expect(
        clippy::mut_from_ref,
        reason = "Field is only accessed here and is guarded by lock with a documented safety comment"
    )]
    fn try_lock<'a>(&'a self) -> Option<(&'a mut Conditions<'sys>, MutexGuard<'a, ExecutorState>)> {
        let guard = self.environment.executor.state.try_lock().ok()?;
        // SAFETY: This is an exclusive access as no other location fetches conditions mutably, and
        // is synchronized by the lock on the executor state.
        let conditions = unsafe { &mut *self.environment.conditions.get() };
        Some((conditions, guard))
    }

    fn tick_executor(&self, allow_inline: bool) -> Option<usize> {
        let mut continuation = None;
        // Ensure that the executor handles any events pushed to the system_completion queue by this thread.
        // If this thread acquires the lock, the executor runs after the push() and they are processed.
        // If this thread does not acquire the lock, then the is_empty() check on the other thread runs
        // after the lock is released, which is after try_lock() failed, which is after the push()
        // on this thread, so the is_empty() check will see the new events and loop.
        loop {
            let Some((conditions, mut guard)) = self.try_lock() else {
                return continuation;
            };
            guard.tick(self, conditions, allow_inline, &mut continuation);
            // Make sure we drop the guard before checking system_completion.is_empty(), or we could lose events.
            drop(guard);
            if self.environment.executor.system_completion.is_empty() {
                return continuation;
            }
        }
    }
}

impl MultiThreadedExecutor {
    /// Reuses a running `Send` task for a bounded chain of ready systems.
    ///
    /// Disabled by default. All access, condition and dependency checks still run.
    /// Independent branches remain available to the pool. A task yields back to
    /// normal dispatch after at most 16 systems; non-Send and exclusive systems
    /// always use their existing thread-affine paths. Ready Send work beyond
    /// the pool/caller capacity remains available for these continuations.
    pub fn with_inline_system_continuations(mut self, enabled: bool) -> Self {
        self.inline_system_continuations = enabled;
        self
    }

    /// Allows otherwise worker-bound systems to run on the schedule's calling thread.
    ///
    /// Disabled by default. When enabled, a `Send` system starting an idle execution
    /// frontier may use the scope executor, including when a worker completes
    /// the last running system. Other systems remain
    /// available to workers, so non-conflicting systems can still run in parallel.
    /// This may reduce handoffs for schedules with short, serialized systems, but
    /// can delay other work serviced by the calling thread when systems run for longer.
    pub fn with_scope_thread_execution(mut self, enabled: bool) -> Self {
        self.scope_thread_execution = enabled;
        self
    }

    /// Creates a new `multi_threaded` executor for use with a [`Schedule`].
    ///
    /// [`Schedule`]: crate::schedule::Schedule
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ExecutorState::new()),
            system_completion: ConcurrentQueue::unbounded(),
            starting_systems: FixedBitSet::new(),
            apply_final_deferred: true,
            scope_thread_execution: false,
            inline_system_continuations: false,
            #[cfg(test)]
            task_spawns: None,
            #[cfg(test)]
            peak_pending_send_tasks: None,
            panic_payload: Mutex::new(None),
            #[cfg(feature = "trace")]
            executor_span: info_span!("multithreaded executor"),
        }
    }
}

impl ExecutorState {
    fn new() -> Self {
        Self {
            system_task_metadata: Vec::new(),
            set_condition_conflicting_systems: Vec::new(),
            num_running_systems: 0,
            num_running_send_systems: 0,
            max_running_send_systems: usize::MAX,
            num_dependencies_remaining: Vec::new(),
            local_thread_running: false,
            exclusive_running: false,
            evaluated_sets: FixedBitSet::new(),
            ready_systems: FixedBitSet::new(),
            ready_systems_copy: FixedBitSet::new(),
            running_systems: FixedBitSet::new(),
            skipped_systems: FixedBitSet::new(),
            completed_systems: FixedBitSet::new(),
            unapplied_systems: FixedBitSet::new(),
        }
    }

    fn tick(
        &mut self,
        context: &Context,
        conditions: &mut Conditions,
        allow_inline: bool,
        continuation: &mut Option<usize>,
    ) {
        #[cfg(feature = "trace")]
        let _span = context.environment.executor.executor_span.enter();

        for result in context.environment.executor.system_completion.try_iter() {
            self.finish_system_and_handle_dependents(result);
        }

        // SAFETY:
        // - `finish_system_and_handle_dependents` has updated the currently running systems.
        // - `rebuild_active_access` locks access for all currently running systems.
        unsafe {
            self.spawn_system_tasks(context, conditions, allow_inline, continuation);
        }
    }

    /// # Safety
    /// - Caller must ensure that `self.ready_systems` does not contain any systems that
    ///   have been mutably borrowed (such as the systems currently running).
    /// - `world_cell` must have permission to access all world data (not counting
    ///   any world data that is claimed by systems currently running on this executor).
    unsafe fn spawn_system_tasks(
        &mut self,
        context: &Context,
        conditions: &mut Conditions,
        allow_inline: bool,
        continuation: &mut Option<usize>,
    ) {
        if self.exclusive_running {
            return;
        }

        #[cfg(feature = "hotpatching")]
        #[expect(
            clippy::undocumented_unsafe_blocks,
            reason = "This actually could result in UB if a system tries to mutate
            `HotPatchChanges`. We allow this as the resource only exists with the `hotpatching` feature.
            and `hotpatching` should never be enabled in release."
        )]
        #[cfg(feature = "hotpatching")]
        let hotpatch_tick = unsafe {
            context
                .environment
                .world_cell
                .get_resource_ref::<HotPatchChanges>()
        }
        .map(|r| r.last_changed())
        .unwrap_or_default();

        // can't borrow since loop mutably borrows `self`
        let mut ready_systems = core::mem::take(&mut self.ready_systems_copy);

        // Skipping systems may cause their dependents to become ready immediately.
        // If that happens, we need to run again immediately or we may fail to spawn those dependents.
        let mut check_for_new_ready_systems = true;
        while check_for_new_ready_systems {
            check_for_new_ready_systems = false;

            ready_systems.clone_from(&self.ready_systems);

            for system_index in ready_systems.ones() {
                debug_assert!(!self.running_systems.contains(system_index));
                // SAFETY: Caller assured that these systems are not running.
                // Therefore, no other reference to this system exists and there is no aliasing.
                let system =
                    &mut unsafe { &mut *context.environment.systems[system_index].get() }.system;

                #[cfg(feature = "hotpatching")]
                if hotpatch_tick.is_newer_than(
                    system.get_last_run(),
                    context.environment.world_cell.change_tick(),
                ) {
                    system.refresh_hotpatch();
                }

                if !self.can_run(system_index, conditions) {
                    // NOTE: exclusive systems with ambiguities are susceptible to
                    // being significantly displaced here (compared to single-threaded order)
                    // if systems after them in topological order can run
                    // if that becomes an issue, `break;` if exclusive system
                    continue;
                }

                self.ready_systems.remove(system_index);

                // SAFETY: `can_run` returned true, which means that:
                // - There can be no systems running whose accesses would conflict with any conditions.
                if unsafe {
                    !self.should_run(
                        system_index,
                        system,
                        conditions,
                        context.environment.world_cell,
                        context.error_handler,
                    )
                } {
                    self.skip_system_and_signal_dependents(system_index);
                    // signal_dependents may have set more systems to ready.
                    check_for_new_ready_systems = true;
                    continue;
                }

                self.running_systems.insert(system_index);
                self.num_running_systems += 1;
                let metadata = &self.system_task_metadata[system_index];
                if metadata.is_send && !metadata.is_exclusive {
                    self.num_running_send_systems += 1;
                }

                if self.system_task_metadata[system_index].is_exclusive {
                    // SAFETY: `can_run` returned true for this system,
                    // which means no systems are currently borrowed.
                    unsafe {
                        self.spawn_exclusive_system_task(context, system_index);
                    }
                    check_for_new_ready_systems = false;
                    break;
                }

                if allow_inline
                    && continuation.is_none()
                    && self.system_task_metadata[system_index].is_send
                {
                    // Access is reserved in running_systems just as for a spawned
                    // task. The caller executes it only after dropping the mutex.
                    *continuation = Some(system_index);
                    continue;
                }

                // SAFETY:
                // - Caller ensured no other reference to this system exists.
                // - `system_task_metadata[system_index].is_exclusive` is `false`,
                //   so `System::is_exclusive` returned `false` when we called it.
                // - `can_run` returned true, so no systems with conflicting world access are running.
                unsafe {
                    self.spawn_system_task(context, system_index);
                }
            }
        }

        // give back
        self.ready_systems_copy = ready_systems;
    }

    fn can_run(&mut self, system_index: usize, conditions: &mut Conditions) -> bool {
        let system_meta = &self.system_task_metadata[system_index];
        if system_meta.is_exclusive && self.num_running_systems > 0 {
            return false;
        }

        if system_meta.is_send
            && !system_meta.is_exclusive
            && self.num_running_send_systems >= self.max_running_send_systems
        {
            return false;
        }

        if !system_meta.is_send && self.local_thread_running {
            return false;
        }

        // TODO: an earlier out if world's archetypes did not change
        for set_idx in conditions.sets_with_conditions_of_systems[system_index]
            .difference(&self.evaluated_sets)
        {
            if !self.set_condition_conflicting_systems[set_idx].is_disjoint(&self.running_systems) {
                return false;
            }
        }

        if !system_meta
            .condition_conflicting_systems
            .is_disjoint(&self.running_systems)
        {
            return false;
        }

        if !self.skipped_systems.contains(system_index)
            && !system_meta
                .conflicting_systems
                .is_disjoint(&self.running_systems)
        {
            return false;
        }

        true
    }

    /// # Safety
    /// * `world` must have permission to read any world data required by
    ///   the system's conditions: this includes conditions for the system
    ///   itself, and conditions for any of the system's sets.
    unsafe fn should_run(
        &mut self,
        system_index: usize,
        system: &mut ScheduleSystem,
        conditions: &mut Conditions,
        world: UnsafeWorldCell,
        error_handler: ErrorHandler,
    ) -> bool {
        let mut should_run = !self.skipped_systems.contains(system_index);

        for set_idx in conditions.sets_with_conditions_of_systems[system_index].ones() {
            if self.evaluated_sets.contains(set_idx) {
                continue;
            }

            // Evaluate the system set's conditions.
            // SAFETY:
            // - The caller ensures that `world` has permission to read any data
            //   required by the conditions.
            let set_conditions_met = unsafe {
                evaluate_and_fold_conditions(
                    &mut conditions.set_conditions[set_idx],
                    world,
                    error_handler,
                    system,
                    true,
                )
            };

            if !set_conditions_met {
                self.skipped_systems
                    .union_with(&conditions.systems_in_sets_with_conditions[set_idx]);
            }

            should_run &= set_conditions_met;
            self.evaluated_sets.insert(set_idx);
        }

        // Evaluate the system's conditions.
        // SAFETY:
        // - The caller ensures that `world` has permission to read any data
        //   required by the conditions.
        let system_conditions_met = unsafe {
            evaluate_and_fold_conditions(
                &mut conditions.system_conditions[system_index],
                world,
                error_handler,
                system,
                false,
            )
        };

        if !system_conditions_met {
            self.skipped_systems.insert(system_index);
        }

        should_run &= system_conditions_met;

        should_run
    }

    /// # Safety
    /// - Caller must not alias systems that are running.
    /// - `is_exclusive` must have returned `false` for the specified system.
    /// - `world` must have permission to access the world data
    ///   used by the specified system.
    unsafe fn spawn_system_task(&mut self, context: &Context, system_index: usize) {
        #[cfg(test)]
        if let Some(count) = &context.environment.executor.task_spawns {
            count.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(test)]
        if let Some(peak) = &context.environment.executor.peak_pending_send_tasks {
            let pending = self
                .running_systems
                .ones()
                .filter(|&index| {
                    let metadata = &self.system_task_metadata[index];
                    metadata.is_send && !metadata.is_exclusive
                })
                .count();
            peak.fetch_max(pending, core::sync::atomic::Ordering::Relaxed);
        }
        // Move the full context object into the new future.
        let context = *context;

        let is_send = self.system_task_metadata[system_index].is_send;
        let budget = if is_send && context.environment.executor.inline_system_continuations {
            MAX_INLINE_SYSTEMS
        } else {
            1
        };

        let task = async move {
            let mut system_index = system_index;
            for completed in 0..budget {
                let system = &context.environment.systems[system_index];
                let res = handle_errors(
                    |system| {
                        // SAFETY: the system was reserved by spawn_system_tasks
                        // after access checks; exclusive systems are never claimed.
                        unsafe {
                            __rust_begin_short_backtrace::run_unsafe(
                                system,
                                context.environment.world_cell,
                            )
                        }
                    },
                    // SAFETY: only this task owns the reserved system.
                    unsafe { &mut (*system.get()).system },
                    context.error_handler,
                    "System panicked",
                );
                let next = context.system_completed(system_index, res, completed + 1 < budget);
                let Some(next) = next else { break };
                system_index = next;
            }
        };

        if is_send {
            if self.num_running_systems == 1 && context.environment.executor.scope_thread_execution
            {
                // Enqueueing does not poll this task inline. Completion is
                // coordinated by the existing scheduling mutex and completion queue.
                context.scope.spawn_on_scope(task);
            } else {
                context.scope.spawn(task);
            }
        } else {
            self.local_thread_running = true;
            context.scope.spawn_on_external(task);
        }
    }

    /// # Safety
    /// Caller must ensure no systems are currently borrowed.
    unsafe fn spawn_exclusive_system_task(&mut self, context: &Context, system_index: usize) {
        let system = &context.environment.systems[system_index];
        // Move the full context object into the new future.
        let context = *context;

        // SAFETY: this system is not running, no other reference exists
        if is_apply_deferred(unsafe { &*(*system.get()).system }) {
            // TODO: avoid allocation
            let unapplied_systems = self.unapplied_systems.clone();
            self.unapplied_systems.clear();
            let task = async move {
                // SAFETY: `can_run` returned true for this system, which means
                // that no other systems currently have access to the world.
                let world = unsafe { context.environment.world_cell.world_mut() };
                let res = apply_deferred(
                    &unapplied_systems,
                    context.environment.systems,
                    world,
                    context.error_handler,
                );
                let _ = context.system_completed(system_index, res, false);
            };

            context.scope.spawn_on_scope(task);
        } else {
            let task = async move {
                // SAFETY: `can_run` returned true for this system, which means
                // that no other systems currently have access to the world.
                let world = unsafe { context.environment.world_cell.world_mut() };
                let res = handle_errors(
                    |system| __rust_begin_short_backtrace::run(system, world),
                    // SAFETY: this system is not running, no other reference exists
                    unsafe { &mut (*system.get()).system },
                    context.error_handler,
                    "Exclusive system panicked",
                );
                let _ = context.system_completed(system_index, res, false);
            };

            context.scope.spawn_on_scope(task);
        }

        self.exclusive_running = true;
        self.local_thread_running = true;
    }

    fn finish_system_and_handle_dependents(&mut self, result: SystemResult) {
        let SystemResult { system_index, .. } = result;

        if self.system_task_metadata[system_index].is_exclusive {
            self.exclusive_running = false;
        }

        if self.system_task_metadata[system_index].is_exclusive
            || !self.system_task_metadata[system_index].is_send
        {
            self.local_thread_running = false;
        }

        debug_assert!(self.num_running_systems >= 1);
        self.num_running_systems -= 1;
        let metadata = &self.system_task_metadata[system_index];
        if metadata.is_send && !metadata.is_exclusive {
            debug_assert!(self.num_running_send_systems >= 1);
            self.num_running_send_systems -= 1;
        }
        self.running_systems.remove(system_index);
        self.completed_systems.insert(system_index);
        self.unapplied_systems.insert(system_index);

        self.signal_dependents(system_index);
    }

    fn skip_system_and_signal_dependents(&mut self, system_index: usize) {
        self.completed_systems.insert(system_index);
        self.signal_dependents(system_index);
    }

    /// Called when `system_index` completes, satisfying one dependency for each of its
    /// dependents and marking any that become ready to run.
    fn signal_dependents(&mut self, system_index: usize) {
        for &dep_idx in &self.system_task_metadata[system_index].dependents {
            let remaining = &mut self.num_dependencies_remaining[dep_idx];
            debug_assert!(*remaining >= 1);
            *remaining -= 1;
            if *remaining == 0 && !self.completed_systems.contains(dep_idx) {
                self.ready_systems.insert(dep_idx);
            }
        }
    }
}

fn apply_deferred(
    unapplied_systems: &FixedBitSet,
    systems: &[SyncUnsafeCell<SystemWithAccess>],
    world: &mut World,
    error_handler: ErrorHandler,
) -> Result<(), Box<dyn Any + Send>> {
    for system_index in unapplied_systems.ones() {
        // SAFETY: none of these systems are running, no other references exist
        let system = &mut unsafe { &mut *systems[system_index].get() }.system;
        handle_errors(
            |system| {
                system.apply_deferred(world);
                Ok(())
            },
            system,
            error_handler,
            "Encountered a panic while applying system buffers",
        )?;
    }
    Ok(())
}

/// # Safety
/// - `world` must have permission to read any world data
///   required by `conditions`.
unsafe fn evaluate_and_fold_conditions(
    conditions: &mut [ConditionWithAccess],
    world: UnsafeWorldCell,
    error_handler: ErrorHandler,
    for_system: &ScheduleSystem,
    on_set: bool,
) -> bool {
    #[expect(
        clippy::unnecessary_fold,
        reason = "Short-circuiting here would prevent conditions from mutating their own state as needed."
    )]
    conditions
        .iter_mut()
        .map(|ConditionWithAccess { condition, .. }| {
            let potential_unwind = std::panic::catch_unwind(AssertUnwindSafe(||
                // SAFETY:
                // - The caller ensures that `world` has permission to read any data
                //   required by the condition.
                unsafe {__rust_begin_short_backtrace::readonly_run_unsafe(&mut **condition, world)}
            ));
            match potential_unwind {
                // Let the error handler handle the panic
                Err(payload) => {
                    __rust_begin_short_backtrace::error_handler(
                        error_handler,
                        BevyError::panic(
                            "Encountered panic",
                            payload,
                        ),
                        ErrorContext::RunCondition {
                            name: condition.name(),
                            last_run: condition.get_last_run(),
                            system: for_system.name(),
                            on_set,
                        },
                    ); false},
                // Condition returned an error, let the error handler handle it
                Ok(Err(RunSystemError::Failed(err))) => {
                    __rust_begin_short_backtrace::error_handler(
                        error_handler,
                        err,
                        ErrorContext::RunCondition {
                            name: condition.name(),
                            last_run: condition.get_last_run(),
                            system: for_system.name(),
                            on_set,
                        },
                    ); false
                },
                Ok(Err(RunSystemError::Skipped(_))) => false,
                Ok(Ok(result)) => result,
            }
        })
        .fold(true, |acc, res| acc && res)
}

/// Handle a potential panic or failed system by invoking the error handler
/// and/or returning a panic payload with which to resume unwinding.
fn handle_errors(
    f: impl FnOnce(&mut BoxedSystem) -> Result<(), RunSystemError>,
    system: &mut BoxedSystem,
    error_handler: ErrorHandler,
    error_message: &str,
) -> Result<(), Box<dyn Any + Send>> {
    let potential_unwind = std::panic::catch_unwind(AssertUnwindSafe(|| f(system)));
    match potential_unwind {
        // Let the error handler handle the panic, passing on any panic it throws
        Err(payload) => std::panic::catch_unwind(AssertUnwindSafe(|| {
            __rust_begin_short_backtrace::error_handler(
                error_handler,
                BevyError::panic(error_message, payload),
                ErrorContext::System {
                    name: system.name(),
                    last_run: system.get_last_run(),
                },
            );
        })),
        // System returned an error, let the error handler handle it, passing on any panic it throws
        Ok(Err(RunSystemError::Failed(err))) => std::panic::catch_unwind(AssertUnwindSafe(|| {
            __rust_begin_short_backtrace::error_handler(
                error_handler,
                err,
                ErrorContext::System {
                    name: system.name(),
                    last_run: system.get_last_run(),
                },
            );
        })),
        // Success (or skipped system)
        _ => Ok(()),
    }
}

/// New-typed [`ThreadExecutor`] [`Resource`] that is used to run systems on the main thread
#[derive(Resource, Clone)]
pub struct MainThreadExecutor(pub Arc<ThreadExecutor<'static>>);

impl Default for MainThreadExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl MainThreadExecutor {
    /// Creates a new executor that can be used to run systems on the main thread.
    pub fn new() -> Self {
        MainThreadExecutor(TaskPool::get_thread_executor())
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use core::{
        panic::AssertUnwindSafe,
        sync::atomic::{AtomicBool, Ordering::Relaxed},
    };
    use std::panic::catch_unwind;

    use crate::{
        change_detection::Tick,
        error::{BevyError, ErrorContext, FallbackErrorHandler},
        prelude::Resource,
        schedule::{IntoScheduleConfigs, MultiThreadedExecutor, Schedule},
        system::{
            Commands, NonSendMut, SystemAccess, SystemMeta, SystemParam, SystemParamValidationError,
        },
        world::{unsafe_world_cell::UnsafeWorldCell, World},
    };

    #[derive(Resource)]
    struct R;

    // T1: observe reserved Send work at the real task-spawn boundary. This
    // bound does not depend on whether a worker or the caller wins the mutex.
    #[test]
    fn inline_continuations_bound_pending_tasks_for_wide_ready_frontiers() {
        use bevy_tasks::{ComputeTaskPool, TaskPool};
        use core::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let workers = ComputeTaskPool::get_or_init(TaskPool::default).thread_num();
        let roots = (workers + 1) * 2 + 16;
        for caller in [false, true] {
            let runs = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let mut executor = MultiThreadedExecutor::new()
                .with_scope_thread_execution(caller)
                .with_inline_system_continuations(true);
            executor.peak_pending_send_tasks = Some(peak.clone());
            let mut schedule = Schedule::default();
            schedule.set_executor(executor);
            for _ in 0..roots {
                let runs = runs.clone();
                schedule.add_systems(move || {
                    runs.fetch_add(1, Relaxed);
                });
            }
            let mut world = World::new();
            for run in 1..=2 {
                peak.store(0, Relaxed);
                schedule.run(&mut world);
                assert_eq!(runs.load(Relaxed), roots * run);
                assert!(
                    peak.load(Relaxed) <= (workers + usize::from(caller)).max(1),
                    "pending Send tasks should fit the available pool/caller capacity"
                );
            }
        }
    }

    // T1: a real Schedule/World and the actual task-spawn boundary. No timing
    // thresholds; independent expected order and the public 16-system budget.
    #[test]
    fn inline_continuations_reuse_tasks_and_yield_at_the_batch_budget() {
        use crate::{prelude::ResMut, schedule::SystemSet};
        use core::sync::atomic::AtomicUsize;
        use std::{sync::Arc, vec::Vec};

        #[derive(Resource, Default)]
        struct Steps(Vec<usize>);
        #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
        struct Step(usize);

        for enabled in [false, true] {
            let count = Arc::new(AtomicUsize::new(0));
            let mut executor =
                MultiThreadedExecutor::new().with_inline_system_continuations(enabled);
            executor.task_spawns = Some(count.clone());
            let mut schedule = Schedule::default();
            schedule.set_executor(executor);
            for index in 0..40 {
                let system = (move |mut steps: ResMut<Steps>| {
                    assert_eq!(steps.0.len() % 40, index);
                    steps.0.push(index);
                })
                .in_set(Step(index));
                if index == 0 {
                    schedule.add_systems(system);
                } else {
                    schedule.add_systems(system.after(Step(index - 1)));
                }
            }
            let mut world = World::new();
            world.init_resource::<Steps>();
            for run in 1..=2 {
                schedule.run(&mut world);
                let expected: Vec<_> = (0..run).flat_map(|_| 0..40).collect();
                assert_eq!(world.resource::<Steps>().0, expected);
                assert_eq!(
                    count.load(Relaxed),
                    if enabled { 3 * run } else { 40 * run },
                    "continuation must reduce real task spawns and release after 16 systems"
                );
            }
        }
    }

    #[test]
    fn inline_continuations_preserve_deferred_exclusive_and_non_send_boundaries() {
        use crate::prelude::{Res, ResMut};
        use std::{thread, vec, vec::Vec};

        #[derive(Resource, Default)]
        struct Trace(Vec<&'static str>);
        #[derive(Default)]
        struct Affinity(Option<thread::ThreadId>);

        for caller in [false, true] {
            let mut world = World::new();
            world.init_resource::<Trace>();
            world.insert_non_send(Affinity::default());
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(caller)
                    .with_inline_system_continuations(true),
            );
            schedule.add_systems(
                (
                    |mut commands: Commands, mut trace: ResMut<Trace>| {
                        trace.0.push("produce");
                        commands.insert_resource(R);
                    },
                    |_: Res<R>, mut trace: ResMut<Trace>| trace.0.push("read"),
                    |world: &mut World| {
                        assert!(world.contains_resource::<R>());
                        world.resource_mut::<Trace>().0.push("exclusive");
                    },
                    |mut affinity: NonSendMut<Affinity>, mut trace: ResMut<Trace>| {
                        affinity.0 = Some(thread::current().id());
                        trace.0.push("non-send");
                    },
                    |mut trace: ResMut<Trace>| trace.0.push("send-after"),
                )
                    .chain(),
            );
            schedule.run(&mut world);
            assert_eq!(
                world.resource::<Trace>().0,
                vec!["produce", "read", "exclusive", "non-send", "send-after"]
            );
            assert_eq!(world.non_send::<Affinity>().0, Some(thread::current().id()));
        }
    }

    #[test]
    fn inline_continuations_finish_dependents_before_propagating_handler_panic() {
        use crate::prelude::ResMut;
        use std::{vec, vec::Vec};

        #[derive(Resource, Default)]
        struct Trace(Vec<usize>);
        fn handler(_: BevyError, _: ErrorContext) {
            panic!("continuation handler panic");
        }

        let mut world = World::new();
        world.init_resource::<Trace>();
        world.insert_resource(FallbackErrorHandler(handler));
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new().with_inline_system_continuations(true));
        schedule.add_systems(
            (
                |mut trace: ResMut<Trace>| trace.0.push(0),
                |mut trace: ResMut<Trace>| {
                    trace.0.push(1);
                    Err(BevyError::ignore("handled error"))
                },
                |mut trace: ResMut<Trace>| trace.0.push(2),
            )
                .chain(),
        );
        let result = catch_unwind(AssertUnwindSafe(|| schedule.run(&mut world)));
        let payload = result.unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"continuation handler panic")
        );
        assert_eq!(world.resource::<Trace>().0, vec![0, 1, 2]);
    }

    #[test]
    fn scope_thread_execution_keeps_a_serial_chain_on_the_caller() {
        for continuations in [false, true] {
            use std::{
                sync::{Arc, Mutex},
                thread, vec,
                vec::Vec,
            };

            let caller = thread::current().id();
            let threads = Arc::new(Mutex::new(Vec::new()));
            let record = || {
                let threads = threads.clone();
                move || threads.lock().unwrap().push(thread::current().id())
            };
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(true)
                    .with_inline_system_continuations(continuations),
            );
            schedule.add_systems((record(), record(), record()).chain());
            schedule.run(&mut World::new());
            assert_eq!(*threads.lock().unwrap(), vec![caller; 3]);
        }
    }

    #[test]
    fn default_executor_keeps_send_systems_on_workers() {
        for continuations in [false, true] {
            use std::{
                sync::{Arc, Mutex},
                thread,
            };

            let caller = thread::current().id();
            let observed = Arc::new(Mutex::new(None));
            let record = observed.clone();
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new().with_inline_system_continuations(continuations),
            );
            schedule.add_systems(move || *record.lock().unwrap() = Some(thread::current().id()));
            schedule.run(&mut World::new());
            assert!(observed
                .lock()
                .unwrap()
                .is_some_and(|thread| thread != caller));
        }
    }

    // T1: readiness is advanced by the last worker after a false condition.
    // The caller is available and the remaining Send work belongs to its scope.
    #[test]
    fn scope_thread_execution_reclaims_worker_ready_frontier() {
        use crate::schedule::SystemSet;
        use std::{
            sync::{mpsc, Arc, Mutex},
            thread,
            time::Duration,
        };

        #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
        struct Root(usize);
        #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
        struct Gate(usize);

        let caller = thread::current().id();
        let (a_tx, a_rx) = mpsc::channel();
        let (b_tx, b_rx) = mpsc::channel();
        let (completed_tx, completed_rx) = mpsc::channel();
        let completed_rx = Arc::new(Mutex::new(completed_rx));
        let make_root = |send: mpsc::Sender<()>, receive: mpsc::Receiver<()>| {
            let receive = Mutex::new(receive);
            let completed_rx = completed_rx.clone();
            move || {
                send.send(()).unwrap();
                receive
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap();
                if thread::current().id() != caller {
                    completed_rx
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(1))
                        .unwrap();
                }
            }
        };
        let condition = || {
            let completed_tx = completed_tx.clone();
            move || {
                // A condition after the caller root can only run once that root
                // is marked complete. The other root therefore completes last.
                completed_tx.send(()).unwrap();
                false
            }
        };
        let observed = Arc::new(Mutex::new(None));
        let record = observed.clone();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new().with_scope_thread_execution(true));
        schedule.add_systems((
            make_root(a_tx, b_rx).in_set(Root(0)),
            make_root(b_tx, a_rx).in_set(Root(1)),
            (|| {}).run_if(condition()).after(Root(0)).in_set(Gate(0)),
            (|| {}).run_if(condition()).after(Root(1)).in_set(Gate(1)),
            (move || *record.lock().unwrap() = Some(thread::current().id()))
                .after(Gate(0))
                .after(Gate(1)),
        ));
        schedule.run(&mut World::new());
        assert_eq!(*observed.lock().unwrap(), Some(caller));
    }

    #[test]
    fn scope_thread_execution_preserves_parallel_work() {
        for continuations in [false, true] {
            use std::{
                sync::{mpsc, Arc, Mutex},
                thread,
                time::Duration,
                vec::Vec,
            };

            let caller = thread::current().id();
            let observed = Arc::new(Mutex::new(Vec::new()));
            let (a_tx, a_rx) = mpsc::channel();
            let (b_tx, b_rx) = mpsc::channel();
            let make_system = |send: mpsc::Sender<()>, receive: mpsc::Receiver<()>| {
                let observed = observed.clone();
                let receive = Mutex::new(receive);
                move || {
                    observed.lock().unwrap().push(thread::current().id());
                    send.send(()).unwrap();
                    receive
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(1))
                        .unwrap();
                }
            };
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(true)
                    .with_inline_system_continuations(continuations),
            );
            // Exercise fan-out from completion, including claiming one inline
            // successor while the other branch must remain available to a worker.
            schedule
                .add_systems((|| {}, (make_system(a_tx, b_rx), make_system(b_tx, a_rx))).chain());
            schedule.run(&mut World::new());
            let observed = observed.lock().unwrap();
            assert_eq!(observed.len(), 2);
            assert!(observed.contains(&caller));
            assert!(observed.iter().any(|thread| *thread != caller));
        }
    }

    #[test]
    fn scope_thread_execution_uses_the_current_run_thread() {
        for continuations in [false, true] {
            use std::{
                sync::{Arc, Mutex},
                thread, vec,
                vec::Vec,
            };

            let observed = Arc::new(Mutex::new(Vec::new()));
            let record = observed.clone();
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(true)
                    .with_inline_system_continuations(continuations),
            );
            schedule.add_systems(move || record.lock().unwrap().push(thread::current().id()));
            let first_caller = thread::current().id();
            let mut world = World::new();
            schedule.run(&mut world);
            let second_caller = thread::spawn(move || {
                schedule.run(&mut world);
                thread::current().id()
            })
            .join()
            .unwrap();
            assert_ne!(first_caller, second_caller);
            assert_eq!(*observed.lock().unwrap(), vec![first_caller, second_caller]);
        }
    }

    struct ExclusiveMarker;

    // SAFETY: No world data is accessed.
    unsafe impl SystemParam for ExclusiveMarker {
        type State = ();
        type Item<'world, 'state> = ExclusiveMarker;

        fn init_state(_world: &mut World) -> Self::State {}

        fn init_access(
            _state: &Self::State,
            system_meta: &mut SystemMeta,
            system_access: &mut SystemAccess,
            _world: &mut World,
        ) {
            system_access.require_exclusive_access::<Self>(system_meta);
        }

        unsafe fn get_param<'world, 'state>(
            _state: &'state mut Self::State,
            _system_meta: &SystemMeta,
            _world: UnsafeWorldCell<'world>,
            _change_tick: Tick,
        ) -> Result<Self::Item<'world, 'state>, SystemParamValidationError> {
            Ok(ExclusiveMarker)
        }
    }

    #[test]
    fn skipped_systems_notify_dependents() {
        for (scope_thread_execution, continuations) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut world = World::new();
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule.add_systems(
                (
                    (|| {}).run_if(|| false),
                    // This system depends on a system that is always skipped.
                    |mut commands: Commands| {
                        commands.insert_resource(R);
                    },
                )
                    .chain(),
            );
            schedule.run(&mut world);
            assert!(world.get_resource::<R>().is_some());
        }
    }

    /// Regression test for case where exclusive system left local thread state as not
    /// cleared and prevented subsequent non-send system runs
    #[test]
    fn exclusive_system_reporting_send_releases_the_local_thread() {
        for (scope_thread_execution, continuations) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            #[derive(Default)]
            struct NonSendMarker(bool);

            let mut world = World::new();
            world.insert_non_send(NonSendMarker::default());

            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule.add_systems(
                (
                    |_: ExclusiveMarker| {},
                    |mut marker: NonSendMut<NonSendMarker>| {
                        marker.0 = true;
                    },
                )
                    .chain(),
            );

            schedule.run(&mut world);
            assert!(world.non_send::<NonSendMarker>().0);
        }
    }

    /// Regression test for a weird bug flagged by MIRI in
    /// `spawn_exclusive_system_task`, related to a `&mut World` being captured
    /// inside an `async` block and somehow remaining alive even after its last use.
    #[test]
    fn check_spawn_exclusive_system_task_miri() {
        for (scope_thread_execution, continuations) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut world = World::new();
            let mut schedule = Schedule::default();
            schedule.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule.add_systems(((|_: Commands| {}), |_: Commands| {}).chain());
            schedule.run(&mut world);
        }
    }

    #[test]
    fn panic_to_error() {
        for (scope_thread_execution, continuations) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut world = World::new();

            let mut schedule_error = Schedule::default();
            schedule_error.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule_error.add_systems(|| Err(BevyError::ignore("")));

            let mut schedule_panic = Schedule::default();
            schedule_panic.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule_panic.add_systems(|| {
                panic!("System's panic payload");
            });

            static HANDLER_CALLED: AtomicBool = AtomicBool::new(false);
            fn handle(_: BevyError, ctx: ErrorContext) {
                assert!(matches!(ctx, ErrorContext::System { .. }));
                HANDLER_CALLED.store(true, Relaxed);
            }
            world.insert_resource(FallbackErrorHandler(handle));

            HANDLER_CALLED.store(false, Relaxed);
            SYSTEM_RAN.store(false, Relaxed);

            // System error
            schedule_error.run(&mut world);
            assert!(HANDLER_CALLED.load(Relaxed));

            // System panic
            HANDLER_CALLED.store(false, Relaxed);
            schedule_panic.run(&mut world);
            assert!(HANDLER_CALLED.load(Relaxed));

            const PANIC_PAYLOAD: &str = "UwU";
            fn panic(_: BevyError, ctx: ErrorContext) {
                assert!(matches!(ctx, ErrorContext::System { .. }));
                panic!("{}", PANIC_PAYLOAD);
            }
            world.insert_resource(FallbackErrorHandler(panic));

            // System error, handler panic
            let result = catch_unwind(AssertUnwindSafe(|| schedule_error.run(&mut world)));
            let payload = result.unwrap_err();
            assert_eq!(
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .unwrap_or_else(|| payload.downcast_ref::<&str>().unwrap()),
                PANIC_PAYLOAD
            );

            // System panic, handler panic
            let result = catch_unwind(AssertUnwindSafe(|| schedule_panic.run(&mut world)));
            let payload = result.unwrap_err();
            assert_eq!(
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .unwrap_or_else(|| payload.downcast_ref::<&str>().unwrap()),
                PANIC_PAYLOAD
            );

            static SYSTEM_RAN: AtomicBool = AtomicBool::new(false);
            let system = || {
                SYSTEM_RAN.store(true, Relaxed);
            };
            let mut schedule_condition_error = Schedule::default();
            schedule_condition_error.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule_condition_error.add_systems(system.run_if(|| Err(BevyError::ignore(""))));

            let mut schedule_condition_panic = Schedule::default();
            schedule_condition_panic.set_executor(
                MultiThreadedExecutor::new()
                    .with_scope_thread_execution(scope_thread_execution)
                    .with_inline_system_continuations(continuations),
            );
            schedule_condition_panic.add_systems(system.run_if(|| {
                panic!("Condition's panic payload");
            }));

            fn handle_condition(_: BevyError, ctx: ErrorContext) {
                assert!(matches!(ctx, ErrorContext::RunCondition { .. }));
                HANDLER_CALLED.store(true, Relaxed);
            }
            fn panic_condition(_: BevyError, ctx: ErrorContext) {
                assert!(matches!(ctx, ErrorContext::RunCondition { .. }));
                HANDLER_CALLED.store(true, Relaxed);
                panic!("{}", PANIC_PAYLOAD);
            }
            world.insert_resource(FallbackErrorHandler(handle_condition));

            // Condition error
            HANDLER_CALLED.store(false, Relaxed);
            SYSTEM_RAN.store(false, Relaxed);
            schedule_condition_error.run(&mut world);
            assert!(HANDLER_CALLED.load(Relaxed));
            assert!(!SYSTEM_RAN.load(Relaxed));

            // Condition panic
            HANDLER_CALLED.store(false, Relaxed);
            SYSTEM_RAN.store(false, Relaxed);
            schedule_condition_panic.run(&mut world);
            assert!(HANDLER_CALLED.load(Relaxed));
            assert!(!SYSTEM_RAN.load(Relaxed));

            world.insert_resource(FallbackErrorHandler(panic_condition));

            // Condition error, handler panic
            HANDLER_CALLED.store(false, Relaxed);
            SYSTEM_RAN.store(false, Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| {
                schedule_condition_error.run(&mut world)
            }));
            let payload = result.unwrap_err();
            assert_eq!(
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .unwrap_or_else(|| payload.downcast_ref::<&str>().unwrap()),
                PANIC_PAYLOAD
            );
            assert!(HANDLER_CALLED.load(Relaxed));
            assert!(!SYSTEM_RAN.load(Relaxed));

            // Condition panic, handler panic
            HANDLER_CALLED.store(false, Relaxed);
            SYSTEM_RAN.store(false, Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| {
                schedule_condition_panic.run(&mut world)
            }));
            let payload = result.unwrap_err();
            assert_eq!(
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .unwrap_or_else(|| payload.downcast_ref::<&str>().unwrap()),
                PANIC_PAYLOAD
            );
            assert!(HANDLER_CALLED.load(Relaxed));
            assert!(!SYSTEM_RAN.load(Relaxed));
        }
    }
}
