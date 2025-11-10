/*! \file
 * \brief GLIF3 alpha synapse implementation
 */

#ifndef _SYNAPSE_TYPES_GLIF3_IMPL_H_
#define _SYNAPSE_TYPES_GLIF3_IMPL_H_

#include <neuron/synapse_types/exp_synapse_utils.h>
#include <debug.h>

#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 4

struct synapse_types_params_t {
    exp_params_t syn_0, syn_1, syn_2, syn_3;
    REAL time_step_ms;
};

struct synapse_types_t {
    exp_state_t syn_0_rise, syn_0, syn_1_rise, syn_1, syn_2_rise, syn_2, syn_3_rise, syn_3;
};

#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

#include <neuron/synapse_types/synapse_types.h>

typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

// Custom init for GLIF3 - uses e/tau instead of standard normalization
static inline void glif3_decay_and_init(exp_state_t *state, exp_params_t *params,
        REAL time_step_ms, uint32_t n_steps_per_timestep) {
    REAL ts = kdivui(time_step_ms, n_steps_per_timestep);
    REAL ts_over_tau = kdivk(ts, params->tau);
    decay_t decay = expulr(-ts_over_tau);

    // TensorFlow line 174: psc_initial = e / tau
    // NOT the standard tau * (1 - exp(-dt/tau))
    REAL e_approx = expulr(ONE);
    decay_t init = kdivk(e_approx, params->tau);

    state->decay = decay;
    state->init = init;
    state->synaptic_input_value = params->init_input;
}

static void synapse_types_initialise(synapse_types_t *s, synapse_types_params_t *p, uint32_t n) {
    glif3_decay_and_init(&s->syn_0_rise, &p->syn_0, p->time_step_ms, n); s->syn_0_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_0, &p->syn_0, p->time_step_ms, n);
    glif3_decay_and_init(&s->syn_1_rise, &p->syn_1, p->time_step_ms, n); s->syn_1_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_1, &p->syn_1, p->time_step_ms, n);
    glif3_decay_and_init(&s->syn_2_rise, &p->syn_2, p->time_step_ms, n); s->syn_2_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_2, &p->syn_2, p->time_step_ms, n);
    glif3_decay_and_init(&s->syn_3_rise, &p->syn_3, p->time_step_ms, n); s->syn_3_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_3, &p->syn_3, p->time_step_ms, n);
}

static void synapse_types_save_state(synapse_types_t *s, synapse_types_params_t *p) {
    p->syn_0.init_input = s->syn_0.synaptic_input_value;
    p->syn_1.init_input = s->syn_1.synaptic_input_value;
    p->syn_2.init_input = s->syn_2.synaptic_input_value;
    p->syn_3.init_input = s->syn_3.synaptic_input_value;
}

static void synapse_types_shape_input(synapse_types_t *p) {
    // Match TensorFlow line 318-319:
    // new_psc_rise = decay * psc_rise + inputs (inputs added by neuron_transfer)
    // new_psc = decay * psc + decay * psc_rise (uses OLD psc_rise before line 318)

    p->syn_0.synaptic_input_value = decay_s1615(p->syn_0.synaptic_input_value, p->syn_0.decay) +
                                     decay_s1615(p->syn_0_rise.synaptic_input_value, p->syn_0.decay);
    exp_shaping(&p->syn_0_rise);

    p->syn_1.synaptic_input_value = decay_s1615(p->syn_1.synaptic_input_value, p->syn_1.decay) +
                                     decay_s1615(p->syn_1_rise.synaptic_input_value, p->syn_1.decay);
    exp_shaping(&p->syn_1_rise);

    p->syn_2.synaptic_input_value = decay_s1615(p->syn_2.synaptic_input_value, p->syn_2.decay) +
                                     decay_s1615(p->syn_2_rise.synaptic_input_value, p->syn_2.decay);
    exp_shaping(&p->syn_2_rise);

    p->syn_3.synaptic_input_value = decay_s1615(p->syn_3.synaptic_input_value, p->syn_3.decay) +
                                     decay_s1615(p->syn_3_rise.synaptic_input_value, p->syn_3.decay);
    exp_shaping(&p->syn_3_rise);
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
