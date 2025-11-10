/*! \file
 * \brief GLIF3 (Generalized Leaky Integrate-and-Fire) Neuron Model
 *
 * Implements GLIF3 model matching TensorFlow reference:
 *   training_code/models.py lines 321-341
 *
 * Continuous-time equations (Allen Institute GLIF specification):
 *   dV/dt = (1/C_m) * [I_syn + I_asc_0 + I_asc_1 - g*(V - E_L)]
 *   dI_asc_j/dt = -k_j * I_asc_j,  j=0,1
 *
 * Discrete-time (TensorFlow lines 325-334):
 *   new_asc_j = exp(-dt*k_j) * asc_j + prev_z * asc_amp_j
 *   reset_current = prev_z * (v_reset - v_th)
 *   new_v = v*decay + current_factor*(I_syn + asc_0 + asc_1 + g*E_L) + reset_current
 *     where prev_z is spike from PREVIOUS timestep (not current!)
 *
 * Exponential Euler integration (TensorFlow lines 169-172):
 *   tau = C_m / g
 *   decay = exp(-dt / tau)
 *   current_factor = (1/C_m) * (1 - decay) * tau = (1 - decay) / g
 *
 * Spike and reset (TensorFlow lines 328, 339-341):
 *   - Voltage NOT hard-reset to V_reset
 *   - Instead: soft reset via reset_current applied NEXT timestep
 *   - ASC amplitudes added NEXT timestep (uses prev_z)
 *   - Refractory timer prevents spiking while active
 *
 * SpiNNaker sub-timestep execution (neuron_impl_standard.h line 257-338):
 *   for i_step in n_steps_per_timestep down to 1:
 *     - Read synapse currents
 *     - Update voltage (this function)
 *     - Check for spike
 *     - If spike: set spiked_last_step=1
 *     - Decay synapses
 *
 * Critical timing with sub-steps:
 *   Problem: With sub-steps, spike in sub-step 1 would apply reset in sub-step 2
 *            of SAME timestep, but TensorFlow uses prev_z (PREVIOUS timestep).
 *   Solution: Only apply reset/ASC on FIRST sub-step of NEXT full timestep.
 */

#ifndef _NEURON_MODEL_GLIF3_IMPL_H_
#define _NEURON_MODEL_GLIF3_IMPL_H_

#include <neuron/models/neuron_model.h>
#include <common/maths-util.h>
#include <debug.h>

// Neuron parameters structure - loaded from SDRAM at startup
struct neuron_params_t {
    // State variables (initial values)
    REAL V_init;              // Initial membrane voltage (mV)
    REAL I_asc_0_init;        // Initial after-spike current 0 (nA)
    REAL I_asc_1_init;        // Initial after-spike current 1 (nA)

    // Model parameters (constant during simulation)
    REAL C_m;                 // Membrane capacitance (nF)
    REAL E_L;                 // Resting potential (mV)
    REAL V_reset;             // Reset voltage (mV)
    REAL V_thresh;            // Threshold voltage (mV)
    REAL asc_amp_0;           // After-spike current amplitude 0 (nA)
    REAL asc_amp_1;           // After-spike current amplitude 1 (nA)
    REAL g;                   // Conductance (uS), g = 1/R
    REAL k0;                  // ASC decay rate 0 (1/ms), k = 1/tau
    REAL k1;                  // ASC decay rate 1 (1/ms)
    REAL t_ref;               // Refractory period (ms)
    REAL I_offset;            // Offset current (nA)
};

// Neuron state structure - stored in DTCM for fast access during simulation
struct neuron_t {
    // Dynamic state variables
    REAL V;                   // Membrane voltage (mV)
    REAL I_asc_0;             // After-spike current 0 (nA)
    REAL I_asc_1;             // After-spike current 1 (nA)

    // Model parameters (copied from params for efficiency)
    REAL C_m;
    REAL E_L;
    REAL V_reset;
    REAL V_thresh;
    REAL asc_amp_0;
    REAL asc_amp_1;
    REAL g;
    REAL k0;
    REAL k1;
    REAL t_ref;
    REAL I_offset;

    // Precomputed exponentials for efficiency
    REAL exp_k0_dt;           // exp(-k0 * dt_full) - ASC decay per FULL timestep
    REAL exp_k1_dt;           // exp(-k1 * dt_full)
    REAL exp_k0_tref;         // exp(-k0 * t_ref) - ASC decay during refractory (unused in current impl)
    REAL exp_k1_tref;         // exp(-k1 * t_ref)
    REAL dt_over_cm;          // dt / C_m (unused with exponential Euler)
    REAL g_dt_over_cm;        // g * dt / C_m (unused with exponential Euler)

    // Exponential Euler integration constants (TensorFlow line 169-172)
    REAL v_decay;             // exp(-dt * g / C_m) = exp(-dt / tau)
    REAL current_factor;      // (1 - v_decay) / g
    REAL reset_current;       // (V_reset - V_thresh) for soft reset

    // Spike tracking for TensorFlow-style delayed reset
    // TensorFlow uses prev_z: spike from PREVIOUS timestep affects CURRENT timestep
    uint32_t spiked_last_step; // 1 if spike occurred in previous full timestep, 0 otherwise

    // Refractory period tracking
    uint32_t refract_timer;   // Countdown timer in sub-timesteps
    uint32_t refract_steps;   // Total refractory period in sub-timesteps

    // Sub-timestep tracking (ensures reset happens only once per full timestep)
    uint32_t n_steps_per_timestep; // Number of sub-steps for integration
    uint32_t sub_step_counter;     // Counts down from n to 1 each full timestep
};

/*! \brief Initialize neuron state and precompute constants
 *
 * Called once at startup (neuron_impl_standard.h line 178)
 *
 * \param[out] state Neuron state structure to initialize
 * \param[in] params Neuron parameters from SDRAM
 * \param[in] n_steps_per_timestep Number of sub-steps for numerical integration
 */
static inline void neuron_model_initialise(neuron_t *state, neuron_params_t *params,
        uint32_t n_steps_per_timestep) {

    // Initialize state variables
    state->V = params->V_init;
    state->I_asc_0 = params->I_asc_0_init;
    state->I_asc_1 = params->I_asc_1_init;

    // Copy parameters to state for fast access
    state->C_m = params->C_m;
    state->E_L = params->E_L;
    state->V_reset = params->V_reset;
    state->V_thresh = params->V_thresh;
    state->asc_amp_0 = params->asc_amp_0;
    state->asc_amp_1 = params->asc_amp_1;
    state->g = params->g;
    state->k0 = params->k0;
    state->k1 = params->k1;
    state->t_ref = params->t_ref;
    state->I_offset = params->I_offset;

    // Sub-timestep duration for integration accuracy
    // With full timestep = 1.0 ms, n=2: dt = 0.5 ms
    REAL dt = 1.0k / (REAL) n_steps_per_timestep;

    // Precompute ASC exponential decays for FULL TIMESTEP
    // TensorFlow line 325-326: exp(-self._dt * k) where self._dt = full timestep
    // CRITICAL: ASC must remain constant during ALL sub-steps (TensorFlow line 333 uses OLD asc),
    //           then decay ONCE at end of full timestep. Using sub-timestep decay causes
    //           amplitude to be incorrectly decayed and voltage to use wrong asc values.
    state->exp_k0_dt = expk(-state->k0 * 1.0k);  // Full timestep decay
    state->exp_k1_dt = expk(-state->k1 * 1.0k);

    // Decay during refractory period (not currently used)
    state->exp_k0_tref = expk(-state->k0 * state->t_ref);
    state->exp_k1_tref = expk(-state->k1 * state->t_ref);

    // Simple integration factors (unused with exponential Euler)
    state->dt_over_cm = dt / state->C_m;
    state->g_dt_over_cm = state->g * dt / state->C_m;

    // Exponential Euler integration constants (TensorFlow line 169-172)
    // tau = C_m / g  (membrane time constant)
    // decay = exp(-dt / tau) = exp(-dt * g / C_m)
    // current_factor = (1/C_m) * (1 - decay) * tau = (1 - decay) / g
    REAL tau = state->C_m / state->g;
    state->v_decay = expk(-dt / tau);
    state->current_factor = (ONE - state->v_decay) / state->g;

    // Soft reset current (TensorFlow line 328)
    // Applied as: V += prev_z * (V_reset - V_thresh)
    // This shifts voltage by (V_reset - V_thresh) when spike occurred last timestep
    state->reset_current = state->V_reset - state->V_thresh;

    // Initialize spike tracking (no spike at t=0)
    state->spiked_last_step = 0;

    // Refractory period in sub-timesteps
    // Example: t_ref=2.0ms, n=2 → refract_steps = 2.0*2 = 4 sub-timesteps
    state->refract_steps = (uint32_t) (state->t_ref * n_steps_per_timestep);
    state->refract_timer = 0;

    // Sub-timestep tracking (starts at n, counts down to 1)
    state->n_steps_per_timestep = n_steps_per_timestep;
    state->sub_step_counter = n_steps_per_timestep;
}

/*! \brief Save neuron state for resumption
 *
 * Called when simulation pauses (neuron.c neuron_pause())
 */
static inline void neuron_model_save_state(neuron_t *state, neuron_params_t *params) {
    params->V_init = state->V;
    params->I_asc_0_init = state->I_asc_0;
    params->I_asc_1_init = state->I_asc_1;
}

/*! \brief Update neuron state for one sub-timestep
 *
 * Implements TensorFlow lines 321-334:
 *   - Decay ASCs: exp(-dt*k) * asc
 *   - Add ASC amps if spike last timestep: + prev_z * asc_amp
 *   - Update voltage: v*decay + current_factor*(I_total + g*E_L)
 *   - Add reset if spike last timestep: + prev_z*(v_reset - v_th)
 *   - Update refractory timer
 *
 * Called once per SUB-STEP (neuron_impl_standard.h line 310)
 *
 * Critical: With sub-steps, must only apply reset/ASC on FIRST sub-step of
 *           NEXT full timestep to match TensorFlow's prev_z semantics.
 *
 * \param[in] num_excitatory_inputs Number of excitatory receptors (4)
 * \param[in] exc_input Array of excitatory synaptic currents
 * \param[in] num_inhibitory_inputs Number of inhibitory receptors (0)
 * \param[in] inh_input Array of inhibitory synaptic currents (unused)
 * \param[in] external_bias External current bias
 * \param[in] current_offset Current injection
 * \param[in,out] neuron Neuron state structure
 * \return Membrane voltage (or V_reset if in refractory period)
 */
static state_t neuron_model_state_update(
        uint16_t num_excitatory_inputs, const input_t* exc_input,
        uint16_t num_inhibitory_inputs, const input_t* inh_input,
        input_t external_bias, REAL current_offset, neuron_t *restrict neuron) {

    // Sub-timestep tracking: detect FIRST and LAST sub-steps by checking BEFORE decrementing
    // Example with n=2: counter starts at 2
    //   Sub-step 1: is_first=true (2==2), is_last=false (2!=1), then decrement to 1
    //   Sub-step 2: is_first=false (1!=2), is_last=true (1==1), then decrement to 0, wrap to 2
    bool is_first_sub_step = (neuron->sub_step_counter == neuron->n_steps_per_timestep);
    bool is_last_sub_step = (neuron->sub_step_counter == 1);

    // Decrement counter (n → n-1 → ... → 1 → n → n-1 ...)
    neuron->sub_step_counter--;
    if (neuron->sub_step_counter == 0) {
        neuron->sub_step_counter = neuron->n_steps_per_timestep;  // Wrap for next timestep
    }

    // Sum all synaptic currents (TensorFlow line 329: input_current = sum(psc))
    REAL total_exc = ZERO;
    REAL total_inh = ZERO;
    for (uint32_t i = 0; i < num_excitatory_inputs; i++) {
        total_exc += exc_input[i];
    }
    for (uint32_t i = 0; i < num_inhibitory_inputs; i++) {
        total_inh += inh_input[i];
    }

    // Total input current including ASCs (TensorFlow line 333: c1 = input + asc_1 + asc_2 + g*E_L)
    // CRITICAL: Use CURRENT asc values (before decay) per Allen Institute GLIF specification
    //           TensorFlow line 333 adds asc_1 and asc_2 from START of timestep
    REAL I_total = total_exc - total_inh + external_bias + neuron->I_offset + current_offset
                   + neuron->I_asc_0 + neuron->I_asc_1;

    // Update voltage using exponential Euler (TensorFlow line 330-334)
    // new_v = v * decay + current_factor * (I_total + g*E_L) + reset_current
    // Where:
    //   decay = exp(-dt/tau)  (line 170: self._decay)
    //   current_factor = (1-decay)/g  (line 172: self._current_factor)
    //   reset_current = prev_z*(v_reset - v_th)  (line 328)
    //
    // Note: Voltage updates DURING refractory period (TensorFlow allows voltage to evolve)
    REAL g_times_EL = neuron->g * neuron->E_L;
    neuron->V = neuron->V * neuron->v_decay +
                neuron->current_factor * (I_total + g_times_EL);

    // Add reset current if spike occurred last FULL timestep (TensorFlow line 328, 334)
    // CRITICAL: Only apply on FIRST sub-step to match TensorFlow's prev_z
    //   TensorFlow: reset_current = prev_z * (v_reset - v_th)
    //               where prev_z is spike from timestep T-1, used in timestep T
    //   With sub-steps: Spike in timestep T must apply reset in timestep T+1, not T
    //                   So only apply when entering NEW full timestep (first sub-step)
    if (neuron->spiked_last_step && is_first_sub_step) {
        neuron->V += neuron->reset_current;
    }

    // Update after-spike currents (TensorFlow line 325-326)
    // new_asc_j = exp(-dt * k_j) * asc_j + prev_z * asc_amp_j
    // Decay happens every sub-step, but amplitude only added on first sub-step
    neuron->I_asc_0 *= neuron->exp_k0_dt;
    neuron->I_asc_1 *= neuron->exp_k1_dt;

    // Add ASC amplitude if spike occurred last FULL timestep (TensorFlow line 325-326)
    // CRITICAL: Only apply on FIRST sub-step to match TensorFlow's prev_z
    //   Prevents adding amplitude multiple times (once per sub-step) for single spike
    if (neuron->spiked_last_step && is_first_sub_step) {
        neuron->I_asc_0 += neuron->asc_amp_0;
        neuron->I_asc_1 += neuron->asc_amp_1;
        neuron->spiked_last_step = 0;  // Clear flag after processing
    }

    // Handle refractory period (TensorFlow line 321, 341)
    // TensorFlow line 321: new_r = relu(r + prev_z*t_ref - dt)
    //   r counts DOWN from t_ref to 0, neurons can't spike while r > 0
    // TensorFlow line 341: new_z = where(new_r > 0, zeros, new_z)
    //   Spike output forced to zero during refractory period
    //
    // Voltage STILL UPDATES during refractory (line 330-334 runs regardless of r)
    // but spike detection is blocked by returning voltage below threshold
    if (neuron->refract_timer > 0) {
        neuron->refract_timer--;  // Countdown in sub-timesteps
        // Return V_reset to prevent spike detection (well below threshold)
        // Threshold check (neuron_impl_standard.h line 317) will see voltage < threshold
        return neuron->V_reset;
    }

    // Return actual voltage for threshold comparison
    return neuron->V;
}

/*! \brief Get membrane voltage for recording
 *
 * \param[in] neuron Neuron state structure
 * \return Current membrane voltage
 */
static state_t neuron_model_get_membrane_voltage(const neuron_t *neuron) {
    return neuron->V;
}

/*! \brief Handle spike occurrence
 *
 * Called when voltage crosses threshold (neuron_impl_standard.h line 324)
 *
 * TensorFlow behavior (lines 328, 325-326):
 *   - Spike flagged as prev_z for NEXT timestep
 *   - Reset applied NEXT timestep: v += prev_z*(v_reset - v_th)
 *   - ASC updated NEXT timestep: asc += prev_z*asc_amp
 *
 * With sub-steps:
 *   - Spike in timestep T, sub-step k: set spiked_last_step=1
 *   - Timestep T+1, sub-step 1: apply reset and ASC (is_first_sub_step=true)
 *   - This ensures one-timestep delay matching TensorFlow's prev_z
 *
 * \param[in,out] neuron Neuron state structure
 */
static void neuron_model_has_spiked(neuron_t *restrict neuron) {
    // Mark that spike occurred (will be prev_z next timestep)
    neuron->spiked_last_step = 1;

    // Start refractory period (TensorFlow line 321: r += prev_z * t_ref)
    neuron->refract_timer = neuron->refract_steps;

    // NOTE: Voltage and ASCs NOT modified here!
    // Reset and ASC amplitude applied in NEXT timestep's state_update
    // (lines 214-216, 238-242) when is_first_sub_step=true
    // This matches TensorFlow's use of prev_z for delayed updates
}

// Debug printing functions
static inline void neuron_model_print_state_variables(const neuron_t *neuron) {
    log_info("V = %11.4k mV", neuron->V);
    log_info("I_asc_0 = %11.4k nA", neuron->I_asc_0);
    log_info("I_asc_1 = %11.4k nA", neuron->I_asc_1);
    log_info("Refract timer = %u", neuron->refract_timer);
}

static inline void neuron_model_print_parameters(const neuron_t *neuron) {
    log_info("C_m = %11.4k nF", neuron->C_m);
    log_info("E_L = %11.4k mV", neuron->E_L);
    log_info("V_reset = %11.4k mV", neuron->V_reset);
    log_info("V_thresh = %11.4k mV", neuron->V_thresh);
    log_info("asc_amp_0 = %11.4k nA", neuron->asc_amp_0);
    log_info("asc_amp_1 = %11.4k nA", neuron->asc_amp_1);
    log_info("g = %11.4k uS", neuron->g);
    log_info("k0 = %11.4k 1/ms", neuron->k0);
    log_info("k1 = %11.4k 1/ms", neuron->k1);
    log_info("t_ref = %11.4k ms", neuron->t_ref);
    log_info("I_offset = %11.4k nA", neuron->I_offset);
}

#endif // _NEURON_MODEL_GLIF3_IMPL_H_
