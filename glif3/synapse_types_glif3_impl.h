/*! \file
 * \brief GLIF3 alpha synapse implementation
 */

#ifndef _SYNAPSE_TYPES_GLIF3_IMPL_H_
#define _SYNAPSE_TYPES_GLIF3_IMPL_H_

#include <neuron/synapse_types/exp_synapse_utils.h>
#include <debug.h>

#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 4

// Parameters for double-exponential synapse (one tau, two init values)
typedef struct double_exp_params_t {
    REAL tau;                 // Time constant (shared by rise and main)
    input_t init_rise;        // Initial value for rise component
    input_t init_main;        // Initial value for main component
} double_exp_params_t;

struct synapse_types_params_t {
    double_exp_params_t syn_0, syn_1, syn_2, syn_3;
    REAL time_step_ms;
};

struct synapse_types_t {
    exp_state_t syn_0_rise, syn_0, syn_1_rise, syn_1, syn_2_rise, syn_2, syn_3_rise, syn_3;
    REAL dt;  // Timestep in ms (needed for TensorFlow line 319 compatibility)
    // Previous timestep's psc_rise values (TensorFlow line 319 uses OLD psc_rise)
    input_t syn_0_rise_prev, syn_1_rise_prev, syn_2_rise_prev, syn_3_rise_prev;
    // Sub-timestep tracking (for correct prev value handling with sub-stepping)
    uint32_t sub_step_counter;
    uint32_t n_steps_per_timestep;
};

#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

#include <neuron/synapse_types/synapse_types.h>

typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

// Helper to init double-exponential synapse with shared tau
static inline void init_double_exp(exp_state_t *rise, exp_state_t *main,
                                    double_exp_params_t *params, REAL dt, uint32_t n) {
    REAL ts = kdivui(dt, n);
    REAL ts_over_tau = kdivk(ts, params->tau);
    decay_t decay = expulr(-ts_over_tau);

    // CRITICAL: TensorFlow uses psc_initial = e / tau (line 174)
    // In TensorFlow line 318, spikes are added AFTER decay: new = old*decay + spike*init
    // But in SpiNNaker, spikes are added by neuron_transfer BEFORE shape_input decays them.
    // This means spikes get decayed in the same timestep, but they shouldn't be.
    // Compensation: multiply init by exp(dt/tau) so after n sub-steps of decay, we get e/tau
    REAL e_approx = expulr(ONE);  // exp(1) = e
    REAL dt_over_tau = kdivk(dt, params->tau);
    REAL compensation = expulr(dt_over_tau);  // exp(dt/tau)
    decay_t init = kdivk(e_approx, params->tau) * compensation;  // (e/tau) * exp(dt/tau)

    // Both rise and main share same decay and init constants
    rise->decay = decay;
    rise->init = init;
    rise->synaptic_input_value = params->init_rise;

    main->decay = decay;
    main->init = init;
    main->synaptic_input_value = params->init_main;
}

static void synapse_types_initialise(synapse_types_t *s, synapse_types_params_t *p, uint32_t n) {
    init_double_exp(&s->syn_0_rise, &s->syn_0, &p->syn_0, p->time_step_ms, n);
    init_double_exp(&s->syn_1_rise, &s->syn_1, &p->syn_1, p->time_step_ms, n);
    init_double_exp(&s->syn_2_rise, &s->syn_2, &p->syn_2, p->time_step_ms, n);
    init_double_exp(&s->syn_3_rise, &s->syn_3, &p->syn_3, p->time_step_ms, n);

    // Store dt for use in shape_input (TensorFlow line 319 compatibility)
    s->dt = p->time_step_ms;

    // Initialize previous psc_rise values to current values
    s->syn_0_rise_prev = s->syn_0_rise.synaptic_input_value;
    s->syn_1_rise_prev = s->syn_1_rise.synaptic_input_value;
    s->syn_2_rise_prev = s->syn_2_rise.synaptic_input_value;
    s->syn_3_rise_prev = s->syn_3_rise.synaptic_input_value;

    // Initialize sub-step tracking
    s->n_steps_per_timestep = n;
    s->sub_step_counter = 0;
}

static void synapse_types_save_state(synapse_types_t *s, synapse_types_params_t *p) {
    // Save all 8 state values (rise and main for each of 4 receptors)
    p->syn_0.init_rise = s->syn_0_rise.synaptic_input_value;
    p->syn_0.init_main = s->syn_0.synaptic_input_value;
    p->syn_1.init_rise = s->syn_1_rise.synaptic_input_value;
    p->syn_1.init_main = s->syn_1.synaptic_input_value;
    p->syn_2.init_rise = s->syn_2_rise.synaptic_input_value;
    p->syn_2.init_main = s->syn_2.synaptic_input_value;
    p->syn_3.init_rise = s->syn_3_rise.synaptic_input_value;
    p->syn_3.init_main = s->syn_3.synaptic_input_value;
}

static void synapse_types_shape_input(synapse_types_t *p) {
    // Match TensorFlow line 319: new_psc = psc * decay + dt * decay * OLD_psc_rise
    // CRITICAL: With sub-stepping, must use ts (sub-timestep) not dt (full timestep)
    // Otherwise contribution gets added N times (once per sub-step) instead of once total
    // psc_rise_prev stays constant across ALL sub-steps (from END of previous FULL timestep)
    REAL dt = p->dt;
    REAL ts = kdivui(dt, p->n_steps_per_timestep);  // Sub-timestep duration

    // Compute new psc using previous FULL timestep's psc_rise values
    // Each sub-step adds ts*decay*psc_rise_prev, totaling to ~dt*decay*psc_rise_prev after N sub-steps
    p->syn_0.synaptic_input_value = decay_s1615(p->syn_0.synaptic_input_value, p->syn_0.decay) +
                                     decay_s1615(ts * p->syn_0_rise_prev, p->syn_0.decay);
    p->syn_1.synaptic_input_value = decay_s1615(p->syn_1.synaptic_input_value, p->syn_1.decay) +
                                     decay_s1615(ts * p->syn_1_rise_prev, p->syn_1.decay);
    p->syn_2.synaptic_input_value = decay_s1615(p->syn_2.synaptic_input_value, p->syn_2.decay) +
                                     decay_s1615(ts * p->syn_2_rise_prev, p->syn_2.decay);
    p->syn_3.synaptic_input_value = decay_s1615(p->syn_3.synaptic_input_value, p->syn_3.decay) +
                                     decay_s1615(ts * p->syn_3_rise_prev, p->syn_3.decay);

    // Decay psc_rise for next sub-step
    exp_shaping(&p->syn_0_rise);
    exp_shaping(&p->syn_1_rise);
    exp_shaping(&p->syn_2_rise);
    exp_shaping(&p->syn_3_rise);

    // Track sub-steps and only save prev values at END of last sub-step
    p->sub_step_counter++;
    if (p->sub_step_counter >= p->n_steps_per_timestep) {
        // Last sub-step: save decayed psc_rise values for next FULL timestep
        p->syn_0_rise_prev = p->syn_0_rise.synaptic_input_value;
        p->syn_1_rise_prev = p->syn_1_rise.synaptic_input_value;
        p->syn_2_rise_prev = p->syn_2_rise.synaptic_input_value;
        p->syn_3_rise_prev = p->syn_3_rise.synaptic_input_value;
        p->sub_step_counter = 0;  // Reset for next full timestep
    }
}

static inline void synapse_types_add_neuron_input(index_t i, synapse_types_t *p, input_t input) {
    exp_state_t* ptrs[4] = {
        &p->syn_0_rise,
        &p->syn_1_rise,
        &p->syn_2_rise,
        &p->syn_3_rise
    };
    add_input_exp(ptrs[i], input);
}

static input_t* synapse_types_get_excitatory_input(input_t *e, synapse_types_t *p) {
    e[0] = p->syn_0.synaptic_input_value;
    e[1] = p->syn_1.synaptic_input_value;
    e[2] = p->syn_2.synaptic_input_value;
    e[3] = p->syn_3.synaptic_input_value;
    return e;
}

static input_t* synapse_types_get_inhibitory_input(input_t *i, synapse_types_t *p) {
    use(i); use(p); return NULL;
}

__attribute__((unused))
static const char *synapse_types_get_type_char(index_t i) { use(i); return ""; }

__attribute__((unused))
static void synapse_types_print_input(synapse_types_t *p) { use(p); }

__attribute__((unused))
static void synapse_types_print_parameters(synapse_types_t *p) { use(p); }

#endif  // _SYNAPSE_TYPES_GLIF3_IMPL_H_
