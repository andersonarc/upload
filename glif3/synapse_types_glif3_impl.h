/*! \file
 * \brief GLIF3 alpha synapse implementation (memory-optimized)
 *
 * Alpha synapse dynamics (Chen et al. 2022, Eq. 3):
 *   C_rise(t+dt) = exp(-dt/tau) * C_rise(t) + (e/tau) * weight * spike
 *   I_syn(t+dt) = exp(-dt/tau) * I_syn(t) + exp(-dt/tau) * C_rise(t)
 */

#ifndef _SYNAPSE_TYPES_GLIF3_IMPL_H_
#define _SYNAPSE_TYPES_GLIF3_IMPL_H_

#include <neuron/synapse_types/exp_synapse_utils.h>
#include <debug.h>

#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 4

struct synapse_types_params_t {
    exp_params_t syn[4];
    REAL time_step_ms;
};

struct synapse_types_t {
    exp_state_t rise[4];
    exp_state_t curr[4];
};

#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

#include <neuron/synapse_types/synapse_types.h>

typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

static inline void synapse_types_initialise(synapse_types_t *state,
        synapse_types_params_t *params, uint32_t n_steps_per_timestep) {
    for (uint32_t i = 0; i < 4; i++) {
        decay_and_init(&state->rise[i], &params->syn[i], params->time_step_ms, n_steps_per_timestep);
        decay_and_init(&state->curr[i], &params->syn[i], params->time_step_ms, n_steps_per_timestep);
        state->curr[i].synaptic_input_value = params->syn[i].init_input;
    }
}

static inline void synapse_types_save_state(synapse_types_t *state,
        synapse_types_params_t *params) {
    for (uint32_t i = 0; i < 4; i++) {
        params->syn[i].init_input = state->curr[i].synaptic_input_value;
    }
}

static inline void synapse_types_shape_input(synapse_types_t *p) {
    for (uint32_t i = 0; i < 4; i++) {
        exp_shaping(&p->rise[i]);
        p->curr[i].synaptic_input_value =
            decay_s1615(p->curr[i].synaptic_input_value, p->curr[i].decay) +
            decay_s1615(p->rise[i].synaptic_input_value, p->curr[i].decay);
    }
}

static inline void synapse_types_add_neuron_input(
        index_t synapse_type_index, synapse_types_t *parameters,
        input_t input) {
    add_input_exp(&parameters->rise[synapse_type_index], input);
}

static inline input_t* synapse_types_get_excitatory_input(
        input_t *excitatory_response, synapse_types_t *parameters) {
    excitatory_response[0] = parameters->curr[0].synaptic_input_value;
    excitatory_response[1] = parameters->curr[1].synaptic_input_value;
    excitatory_response[2] = parameters->curr[2].synaptic_input_value;
    excitatory_response[3] = parameters->curr[3].synaptic_input_value;
    return &excitatory_response[0];
}

static inline input_t* synapse_types_get_inhibitory_input(
        input_t *inhibitory_response, synapse_types_t *parameters) {
    use(inhibitory_response);
    use(parameters);
    return NULL;
}

static inline const char *synapse_types_get_type_char(index_t synapse_type_index) {
    use(synapse_type_index);
    return "0";
}

static inline void synapse_types_print_input(synapse_types_t *parameters) {
    use(parameters);
}

static inline void synapse_types_print_parameters(synapse_types_t *parameters) {
    use(parameters);
}

#endif  // _SYNAPSE_TYPES_GLIF3_IMPL_H_
