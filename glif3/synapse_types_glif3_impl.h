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
    // NOT the standard exponential synapse normalization tau * (1 - exp(-dt/tau))
    // e ≈ 2.71828, using expulr(1.0) to get e in fixed-point
    REAL e_approx = expulr(ONE);  // exp(1) = e
    decay_t init = kdivk(e_approx, params->tau);  // e / tau (matches TensorFlow)

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
    // CRITICAL: psc_rise_prev holds value from END of previous timestep (after decay, before new spikes added)
    // TensorFlow line 318-319 compute new values using OLD values on RHS
    REAL dt = p->dt;

    // Compute new psc using previous timestep's psc_rise values (before neuron_transfer added current spikes)
    p->syn_0.synaptic_input_value = decay_s1615(p->syn_0.synaptic_input_value, p->syn_0.decay) +
                                     decay_s1615(dt * p->syn_0_rise_prev, p->syn_0.decay);
    p->syn_1.synaptic_input_value = decay_s1615(p->syn_1.synaptic_input_value, p->syn_1.decay) +
                                     decay_s1615(dt * p->syn_1_rise_prev, p->syn_1.decay);
    p->syn_2.synaptic_input_value = decay_s1615(p->syn_2.synaptic_input_value, p->syn_2.decay) +
                                     decay_s1615(dt * p->syn_2_rise_prev, p->syn_2.decay);
    p->syn_3.synaptic_input_value = decay_s1615(p->syn_3.synaptic_input_value, p->syn_3.decay) +
                                     decay_s1615(dt * p->syn_3_rise_prev, p->syn_3.decay);

    // Decay psc_rise for next timestep
    exp_shaping(&p->syn_0_rise);
    exp_shaping(&p->syn_1_rise);
    exp_shaping(&p->syn_2_rise);
    exp_shaping(&p->syn_3_rise);

    // Save decayed psc_rise values to prev for next timestep (AFTER decay, ready for next timestep)
    p->syn_0_rise_prev = p->syn_0_rise.synaptic_input_value;
    p->syn_1_rise_prev = p->syn_1_rise.synaptic_input_value;
    p->syn_2_rise_prev = p->syn_2_rise.synaptic_input_value;
    p->syn_3_rise_prev = p->syn_3_rise.synaptic_input_value;
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
