import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import arviz as az

accepted = 0

# Set random seed for reproducibility
np.random.seed(42)

# 1. DEFINITION OF THE ANALYTICAL SOLUTION (THE TARGET)
# =============================================================================
# From Assignment 8 and the new data
historical_estimate = 0.345
k_obs = 7  # number of successes
n_obs = 20  # number of trials

#  posterior (Beta distribution)
alpha_prior = 34.5  # chosen so that mean = 34.5/100 = 0.345
beta_prior = 65.5
alpha_post = alpha_prior + k_obs
beta_post = beta_prior + n_obs - k_obs

analytical_posterior = stats.beta(alpha_post, beta_post)


# 2. METROPOLIS-HASTINGS ALGORITHM IMPLEMENTATION
# =============================================================================
def metropolis_hastings(n_iterations=10000, burn_in=1000, delta=0.1):
    """Metropolis-Hastings algorithm for Beta-Binomial model."""
    # Initialize
    psi_current = np.random.uniform(0, 1)
    samples = np.zeros(n_iterations)
    accepted = 0
    
    for i in range(n_iterations):
        # Propose new value (uniform random walk)
        psi_proposed = psi_current + np.random.uniform(-delta, delta)
        
        # Handle boundary conditions (reflect proposals outside [0,1])
        if psi_proposed < 0:
            psi_proposed = -psi_proposed
        elif psi_proposed > 1:
            psi_proposed = 2 - psi_proposed
        
        # Calculate acceptance ratio (using log to avoid underflow)
        log_current = analytical_posterior.logpdf(psi_current)
        log_proposed = analytical_posterior.logpdf(psi_proposed)
        acceptance_ratio = np.exp(log_proposed - log_current)
        
        # Accept or reject
        if np.random.random() < acceptance_ratio:
            psi_current = psi_proposed
            accepted += 1
        
        samples[i] = psi_current
    
    acceptance_rate = accepted / n_iterations
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    return samples[burn_in:]

# Run the algorithm
posterior_samples = metropolis_hastings()

# =============================================================================
# 3. CONVERGENCE DIAGNOSTICS
# =============================================================================
# For Gelman-Rubin, we need multiple chains
print("\nRunning multiple chains for convergence diagnostics...")
n_chains = 4
all_chains = []

for chain in range(n_chains):
    np.random.seed(42 + chain)  # Different seed for each chain
    chain_samples = metropolis_hastings()
    all_chains.append(chain_samples)

all_chains_array = np.array(all_chains)

# Calculate R-hat
r_hat = az.rhat(all_chains_array)
print(f"Gelman-Rubin R-hat: {r_hat:.3f}")

# 4. COMPARE MCMC WITH ANALYTICAL SOLUTION
# =============================================================================
# Calculate summary statistics
mcmc_mean = np.mean(posterior_samples)
mcmc_ci = np.percentile(posterior_samples, [2.5, 97.5])

analytical_mean = analytical_posterior.mean()
analytical_ci = analytical_posterior.interval(0.95)

print(f"\nComparison of Results:")
print(f"{'':<20} {'Mean':<10} {'95% CI':<20}")
print(f"{'Analytical':<20} {analytical_mean:.4f}   ({analytical_ci[0]:.3f}, {analytical_ci[1]:.3f})")
print(f"{'MCMC':<20} {mcmc_mean:.4f}   ({mcmc_ci[0]:.3f}, {mcmc_ci[1]:.3f})")

# 5. CREATE DIAGNOSTIC PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Trace plot
axes[0,0].plot(posterior_samples, alpha=0.7)
axes[0,0].set_xlabel('Iteration')
axes[0,0].set_ylabel('ψ')
axes[0,0].set_title('Trace Plot: MCMC Samples')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Autocorrelation
az.plot_autocorr(all_chains_array, ax=axes[0,1])
axes[0,1].set_title('Autocorrelation')

# Plot 3: Posterior distribution comparison
x = np.linspace(0.2, 0.5, 1000)
axes[1,0].hist(posterior_samples, bins=50, density=True, alpha=0.7, 
               label='MCMC Samples', color='skyblue')
axes[1,0].plot(x, analytical_posterior.pdf(x), 'r-', lw=2, 
               label=f'Analytical Beta({alpha_post},{beta_post})')
axes[1,0].set_xlabel('Co-occurrence Probability ψ')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Posterior Distribution')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Running mean
running_mean = np.cumsum(posterior_samples) / np.arange(1, len(posterior_samples) + 1)
axes[1,1].plot(running_mean)
axes[1,1].axhline(analytical_mean, color='r', linestyle='--', label='Analytical Mean')
axes[1,1].set_xlabel('Iteration')
axes[1,1].set_ylabel('Running Mean of ψ')
axes[1,1].set_title('Convergence of Mean Estimate')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle(f'MCMC Diagnostics for Co-Occurrence Probability ψ\n'
             f'R-hat = {r_hat:.3f}, Acceptance Rate = {accepted/10000:.3f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()


# 6. FINAL RESULTS TABLE
# =============================================================================
print(f"\n{'='*50}")
print("FINAL RESULTS FOR MINISTRY DECISION-MAKING")
print(f"{'='*50}")
print(f"Historical prior estimate (Assignment 8): {historical_estimate:.3f}")
print(f"New surveillance data: {k_obs} events in {n_obs} days ({k_obs/n_obs:.3f})")
print(f"Posterior mean estimate: {mcmc_mean:.3f}")
print(f"95% Credible Interval: ({mcmc_ci[0]:.3f}, {mcmc_ci[1]:.3f})")
print(f"{'='*50}")