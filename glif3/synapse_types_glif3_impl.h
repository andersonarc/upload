/*! \file
 * \brief GLIF3 alpha synapse implementation
 *
 * Implements double-exponential (alpha) synapses matching TensorFlow reference:
 *   training_code/models.py lines 318-319
 *
 * Continuous-time equations:
 *   d(psc_rise)/dt = -psc_rise/tau + spike_inputs(t) * psc_initial
 *   d(psc)/dt = -psc/tau + psc_rise
 *
 * Discrete-time (TensorFlow line 318-319):
 *   new_psc_rise = exp(-dt/tau) * psc_rise + spike_inputs * psc_initial
 *   new_psc = exp(-dt/tau) * psc + dt * exp(-dt/tau) * psc_rise_OLD
 *     where psc_rise_OLD is value BEFORE spike_inputs added
 *
 * NOTE: TensorFlow has "dt * decay * psc_rise" but our implementation uses
 *       "decay * psc_rise" (no dt factor). With sub-timesteps, adding dt per
 *       sub-step causes N× over-contribution. Numerical integration via sub-steps
 *       provides accurate evolution without explicit dt multiplication
 *
 * SpiNNaker execution order (see neuron.c, neuron_impl_standard.h):
 *   1. neuron_transfer() - adds spike_inputs to psc_rise (ONCE per timestep)
 *   2. Sub-step loop (n_steps_per_timestep iterations):
 *      a. Read psc values via synapse_types_get_excitatory_input()
 *      b. Update neuron voltage
 *      c. Check for spike
 *      d. Call synapse_types_shape_input() - decay psc and psc_rise
 *
 * Critical timing issue:
 *   neuron_transfer adds inputs BEFORE sub-steps run, but TensorFlow uses
 *   psc_rise from BEFORE inputs. Solution: save psc_rise_prev at end of
 *   previous timestep, use saved value in psc calculation.
 */

#ifndef _SYNAPSE_TYPES_GLIF3_IMPL_H_
#define _SYNAPSE_TYPES_GLIF3_IMPL_H_

#include <neuron/synapse_types/exp_synapse_utils.h>
#include <debug.h>

#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 4

struct synapse_types_params_t {
    exp_params_t syn_0, syn_1, syn_2, syn_3;
    REAL time_step_ms;  // Full simulation timestep (e.g., 1.0 ms)
};

struct synapse_types_t {
    // Synapse state: psc_rise and psc for each of 4 receptors
    exp_state_t syn_0_rise, syn_0, syn_1_rise, syn_1, syn_2_rise, syn_2, syn_3_rise, syn_3;

    REAL dt;  // Full timestep duration (matches TensorFlow self._dt)

    // Previous timestep psc_rise values (saved at END of timestep T-1, used during timestep T)
    // Required because neuron_transfer adds inputs BEFORE sub-steps, but TensorFlow
    // uses psc_rise from BEFORE inputs (line 319: uses psc_rise before line 318 adds inputs)
    REAL syn_0_rise_prev, syn_1_rise_prev, syn_2_rise_prev, syn_3_rise_prev;

    // Sub-timestep tracking for proper timing with SpiNNaker sub-stepping
    uint32_t n_steps_per_timestep;  // Number of sub-steps per full timestep
    uint32_t sub_step_counter;      // Counts down from n to 1 each full timestep
};

#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

#include <neuron/synapse_types/synapse_types.h>

typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

/*! \brief Initialize synapse decay and psc_initial for GLIF3
 *
 * TensorFlow line 174: psc_initial = e / tau
 * This differs from standard SpiNNaker normalization: tau * (1 - exp(-dt/tau))
 *
 * For tau=5ms, dt=1ms:
 *   TensorFlow: e/5 = 0.544
 *   Standard SpiNNaker: 5*(1-exp(-1/5)) = 0.91
 * Using standard normalization causes ~67% stronger inputs than TensorFlow.
 *
 * Sub-timestep decay:
 *   ts = time_step_ms / n_steps_per_timestep  (e.g., 1.0/2 = 0.5 ms)
 *   decay = exp(-ts/tau)  (e.g., exp(-0.5/5) = 0.9048 per sub-step)
 *   Full timestep: 0.9048^2 = 0.8187 ≈ exp(-1.0/5) ✓
 *
 * \param[out] state The synapse state to initialize
 * \param[in] params The synapse parameters (tau, init_input)
 * \param[in] time_step_ms Full simulation timestep in milliseconds
 * \param[in] n_steps_per_timestep Number of sub-steps for numerical integration
 */
static inline void glif3_decay_and_init(exp_state_t *state, exp_params_t *params,
        REAL time_step_ms, uint32_t n_steps_per_timestep) {
    // Sub-timestep duration for accurate integration
    REAL ts = kdivui(time_step_ms, n_steps_per_timestep);
    REAL ts_over_tau = kdivk(ts, params->tau);

    // Exponential decay per sub-timestep: exp(-ts/tau)
    decay_t decay = expulr(-ts_over_tau);

    // TensorFlow line 174: psc_initial = e / tau
    // NOT standard SpiNNaker: tau * (1 - exp(-dt/tau))
    REAL e_approx = expulr(ONE);  // Approximate e ≈ 2.718
    decay_t init = kdivk(e_approx, params->tau);

    state->decay = decay;
    state->init = init;
    state->synaptic_input_value = params->init_input;
}

/*! \brief Initialize all synapse state and sub-timestep tracking
 *
 * Called once at startup (see neuron_impl_standard.h line 178-189)
 *
 * \param[out] s Synapse state structure to initialize
 * \param[in] p Synapse parameters from SDRAM
 * \param[in] n Number of sub-steps per timestep
 */
static void synapse_types_initialise(synapse_types_t *s, synapse_types_params_t *p, uint32_t n) {
    // Store full timestep duration (TensorFlow self._dt, typically 1.0 ms)
    s->dt = p->time_step_ms;

    // Initialize all 4 receptors with GLIF3-specific normalization
    glif3_decay_and_init(&s->syn_0_rise, &p->syn_0, p->time_step_ms, n);
    s->syn_0_rise.synaptic_input_value = 0.0k;  // psc_rise starts at zero
    glif3_decay_and_init(&s->syn_0, &p->syn_0, p->time_step_ms, n);

    glif3_decay_and_init(&s->syn_1_rise, &p->syn_1, p->time_step_ms, n);
    s->syn_1_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_1, &p->syn_1, p->time_step_ms, n);

    glif3_decay_and_init(&s->syn_2_rise, &p->syn_2, p->time_step_ms, n);
    s->syn_2_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_2, &p->syn_2, p->time_step_ms, n);

    glif3_decay_and_init(&s->syn_3_rise, &p->syn_3, p->time_step_ms, n);
    s->syn_3_rise.synaptic_input_value = 0.0k;
    glif3_decay_and_init(&s->syn_3, &p->syn_3, p->time_step_ms, n);

    // Initialize psc_rise_prev to zero (no previous spikes at t=0)
    s->syn_0_rise_prev = 0.0k;
    s->syn_1_rise_prev = 0.0k;
    s->syn_2_rise_prev = 0.0k;
    s->syn_3_rise_prev = 0.0k;

    // Sub-timestep tracking (see neuron_impl_standard.h line 257 for loop structure)
    s->n_steps_per_timestep = n;
    s->sub_step_counter = n;  // Starts at n, counts down to 1
}

/*! \brief Save synapse state for resumption
 *
 * Called when simulation pauses (see neuron.c neuron_pause())
 */
static void synapse_types_save_state(synapse_types_t *s, synapse_types_params_t *p) {
    p->syn_0.init_input = s->syn_0.synaptic_input_value;
    p->syn_1.init_input = s->syn_1.synaptic_input_value;
    p->syn_2.init_input = s->syn_2.synaptic_input_value;
    p->syn_3.init_input = s->syn_3.synaptic_input_value;
}

/*! \brief Update synapse state: decay psc and psc_rise
 *
 * Implements TensorFlow line 318-319:
 *   new_psc_rise = decay * psc_rise + inputs
 *   new_psc = decay * psc + decay * psc_rise_OLD
 *
 * Called AFTER neuron voltage update, once per SUB-STEP
 * (see neuron_impl_standard.h line 337)
 *
 * Critical timing with sub-steps:
 *   Execution order per FULL timestep:
 *     1. neuron_transfer() adds spike inputs to psc_rise (ONCE)
 *     2. Sub-step 1: reads psc (has OLD psc_rise), decays both
 *     3. Sub-step 2: reads psc (has decayed psc_rise from step 1), decays both
 *     ...
 *
 *   Problem: After neuron_transfer, psc_rise has NEW inputs, but TensorFlow
 *            uses OLD psc_rise (before inputs) in psc calculation.
 *
 *   Solution: Save psc_rise at END of timestep T-1 as psc_rise_prev,
 *            use psc_rise_prev throughout timestep T.
 *
 * \param[in,out] p Synapse state structure
 */
static void synapse_types_shape_input(synapse_types_t *p) {
    // Sub-timestep tracking: check counter BEFORE decrementing
    bool is_first_sub_step = (p->sub_step_counter == p->n_steps_per_timestep);
    bool is_last_sub_step = (p->sub_step_counter == 1);

    // Decrement counter (n → n-1 → ... → 1 → n → n-1 ...)
    p->sub_step_counter--;
    if (p->sub_step_counter == 0) {
        p->sub_step_counter = p->n_steps_per_timestep;  // Wrap to n for next timestep
    }

    // TensorFlow line 319: new_psc = psc*decay + dt*decay*psc_rise_OLD
    // CRITICAL: Add psc_rise contribution ONCE per full timestep (first sub-step only)
    //           Otherwise with n sub-steps, total contribution is multiplied by ~n
    //
    // Decay psc every sub-step
    p->syn_0.synaptic_input_value = decay_s1615(p->syn_0.synaptic_input_value, p->syn_0.decay);
    // Add psc_rise_OLD contribution only on first sub-step
    if (is_first_sub_step) {
        p->syn_0.synaptic_input_value += p->dt * decay_s1615(p->syn_0_rise_prev, p->syn_0.decay);
    }

    // TensorFlow line 318: new_psc_rise = decay*psc_rise + inputs
    // Inputs already added by neuron_transfer, exp_shaping just does decay part
    exp_shaping(&p->syn_0_rise);

    // Repeat for other 3 receptors
    p->syn_1.synaptic_input_value = decay_s1615(p->syn_1.synaptic_input_value, p->syn_1.decay);
    if (is_first_sub_step) {
        p->syn_1.synaptic_input_value += p->dt * decay_s1615(p->syn_1_rise_prev, p->syn_1.decay);
    }
    exp_shaping(&p->syn_1_rise);

    p->syn_2.synaptic_input_value = decay_s1615(p->syn_2.synaptic_input_value, p->syn_2.decay);
    if (is_first_sub_step) {
        p->syn_2.synaptic_input_value += p->dt * decay_s1615(p->syn_2_rise_prev, p->syn_2.decay);
    }
    exp_shaping(&p->syn_2_rise);

    p->syn_3.synaptic_input_value = decay_s1615(p->syn_3.synaptic_input_value, p->syn_3.decay);
    if (is_first_sub_step) {
        p->syn_3.synaptic_input_value += p->dt * decay_s1615(p->syn_3_rise_prev, p->syn_3.decay);
    }
    exp_shaping(&p->syn_3_rise);

    // On LAST sub-step of timestep T: save psc_rise AFTER decay for use in timestep T+1
    // CRITICAL: Save AFTER exp_shaping so psc_rise_prev has END of timestep value
    // Timing: This captures psc_rise AFTER all sub-steps have decayed it, but BEFORE
    //         next timestep's neuron_transfer adds new inputs.
    // Result: psc_rise_prev = value at END of timestep T, used in timestep T+1.
    //         Matches TensorFlow's use of psc_rise from BEFORE inputs added (line 319).
    if (is_last_sub_step) {
        p->syn_0_rise_prev = p->syn_0_rise.synaptic_input_value;
        p->syn_1_rise_prev = p->syn_1_rise.synaptic_input_value;
        p->syn_2_rise_prev = p->syn_2_rise.synaptic_input_value;
        p->syn_3_rise_prev = p->syn_3_rise.synaptic_input_value;
    }
}

/*! \brief Add spike input to synapse
 *
 * Called by neuron_transfer() which processes ring buffer
 * (see neuron.c line 204-227)
 *
 * Adds weighted spike to psc_rise, implements input term in TensorFlow line 318.
 * Note: Scaling by psc_initial happens in add_input_exp() via state->init.
 *
 * \param[in] i Synapse index (0-3)
 * \param[in,out] p Synapse state structure
 * \param[in] input Weighted spike input
 */
static inline void synapse_types_add_neuron_input(index_t i, synapse_types_t *p, input_t input) {
    exp_state_t* ptrs[4] = {
        &p->syn_0_rise,
        &p->syn_1_rise,
        &p->syn_2_rise,
        &p->syn_3_rise
    };
    // add_input_exp (exp_synapse_utils.h line 70) adds: input * state->init
    // where state->init = e/tau from glif3_decay_and_init()
    add_input_exp(ptrs[i], input);
}

/*! \brief Get excitatory synaptic currents
 *
 * Returns psc values (NOT psc_rise) as these represent actual current input.
 * TensorFlow line 329: input_current = tf.reduce_sum(psc, -1)
 *
 * Called each sub-step before voltage update (neuron_impl_standard.h line 264)
 *
 * \param[out] e Array to fill with 4 excitatory receptor currents
 * \param[in] p Synapse state structure
 * \return Pointer to filled array
 */
static input_t* synapse_types_get_excitatory_input(input_t *e, synapse_types_t *p) {
    e[0] = p->syn_0.synaptic_input_value;  // psc, not psc_rise
    e[1] = p->syn_1.synaptic_input_value;
    e[2] = p->syn_2.synaptic_input_value;
    e[3] = p->syn_3.synaptic_input_value;
    return e;
}

// No inhibitory receptors in GLIF3
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
