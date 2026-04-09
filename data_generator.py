"""
Synthetic Patient Data Generator
=================================
Generates realistic, heterogeneous patient datasets for each hospital node.
Each hospital has a different data distribution (non-IID) to simulate
real-world federated learning challenges.
"""

import os
import numpy as np
import pandas as pd
import config


def generate_hospital_data(hospital_config, seed=None):
    """
    Generate synthetic patient data for a single hospital.
    
    Each hospital has distinct demographic distributions to create
    non-IID (non-identically distributed) data across the federation.
    
    Args:
        hospital_config: Dictionary with hospital-specific parameters
        seed: Random seed for reproducibility
        
    Returns:
        pandas DataFrame with patient records
    """
    if seed is not None:
        np.random.seed(seed)

    n = hospital_config['num_samples']
    outbreak_rate = hospital_config['outbreak_rate']
    age_mean = hospital_config['age_mean']
    age_std = hospital_config['age_std']
    hospital_id = hospital_config['id']

    # ── Demographics ──────────────────────────────────────────
    age = np.clip(np.random.normal(age_mean, age_std, n), 1, 100).astype(int)
    gender = np.random.binomial(1, 0.48, n)  # 0=Female, 1=Male
    region_code = np.random.randint(0, 10, n)  # Regional coding

    # ── Clinical Features ─────────────────────────────────────
    # Symptom severity (0-10 scale), correlated with age
    base_severity = np.random.exponential(2.0, n)
    age_factor = (age - 30) / 50  # Older → higher severity
    symptom_severity = np.clip(base_severity + age_factor, 0, 10)

    # Body temperature (°F), slightly elevated in symptomatic patients
    body_temperature = np.random.normal(98.6, 0.8, n)
    body_temperature += symptom_severity * 0.15  # Correlation with symptoms

    # ── Epidemiological Features ──────────────────────────────
    # Contact count: number of recent contacts with infected individuals
    contact_count = np.random.poisson(3 + hospital_id * 0.5, n)

    # Vaccination status (0=unvaccinated, 1=vaccinated)
    vax_rate = 0.6 + hospital_id * 0.05  # Varies by hospital/region
    vaccination_status = np.random.binomial(1, min(vax_rate, 0.95), n)

    # Comorbidity index (0-5, Charlson-like)
    comorbidity_index = np.clip(
        np.random.poisson(0.5 + age / 60, n), 0, 5
    )

    # Population density of patient's area (people per sq km, normalized)
    population_density = np.random.lognormal(
        mean=5 + hospital_id * 0.3, sigma=0.8, size=n
    )
    population_density = np.clip(population_density / population_density.max(), 0, 1)

    # Travel history (0=no recent travel, 1=recent international travel)
    travel_rate = 0.1 + hospital_id * 0.03
    travel_history = np.random.binomial(1, min(travel_rate, 0.4), n)

    # Days since symptom onset
    days_since_symptoms = np.clip(
        np.random.exponential(5, n) * (symptom_severity > 2), 0, 30
    ).astype(int)

    # Hospitalization history (0=none, 1=previously hospitalized)
    hosp_rate = 0.05 + comorbidity_index * 0.05
    hospitalization_history = np.random.binomial(1, np.clip(hosp_rate, 0, 0.5))

    # ── Outbreak Label ────────────────────────────────────────
    # Generate label based on risk factors (logistic model)
    log_odds = (
        -2.0
        + 0.03 * age
        + 0.4 * symptom_severity
        + 0.2 * (body_temperature - 98.6)
        + 0.15 * contact_count
        - 0.8 * vaccination_status
        + 0.3 * comorbidity_index
        + 0.5 * population_density
        + 0.4 * travel_history
        + 0.05 * days_since_symptoms
        + 0.3 * hospitalization_history
    )

    # Adjust intercept to match desired outbreak rate
    prob = 1 / (1 + np.exp(-log_odds))
    adjustment = np.log(outbreak_rate / (1 - outbreak_rate)) - np.log(
        prob.mean() / (1 - prob.mean())
    )
    adjusted_prob = 1 / (1 + np.exp(-(log_odds + adjustment)))
    outbreak_risk = np.random.binomial(1, np.clip(adjusted_prob, 0.01, 0.99))

    # ── Assemble DataFrame ────────────────────────────────────
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'region_code': region_code,
        'symptom_severity': np.round(symptom_severity, 2),
        'body_temperature': np.round(body_temperature, 2),
        'contact_count': contact_count,
        'vaccination_status': vaccination_status,
        'comorbidity_index': comorbidity_index,
        'population_density': np.round(population_density, 4),
        'travel_history': travel_history,
        'days_since_symptoms': days_since_symptoms,
        'hospitalization_history': hospitalization_history,
        'outbreak_risk': outbreak_risk,
    })

    return df


def generate_all_hospital_data(hospitals=None, base_seed=42):
    """
    Generate synthetic data for all hospitals and save to disk.
    
    Args:
        hospitals: List of hospital configurations (uses config default if None)
        base_seed: Base random seed (each hospital gets base_seed + id)
        
    Returns:
        Dictionary mapping hospital_id → DataFrame
    """
    if hospitals is None:
        hospitals = config.HOSPITALS

    all_data = {}

    print("\n" + "=" * 60)
    print("  SYNTHETIC PATIENT DATA GENERATION")
    print("=" * 60)

    for hosp in hospitals:
        hosp_id = hosp['id']
        hosp_dir = os.path.join(config.DATA_DIR, f"hospital_{hosp_id}")
        os.makedirs(hosp_dir, exist_ok=True)

        df = generate_hospital_data(hosp, seed=base_seed + hosp_id)
        csv_path = os.path.join(hosp_dir, "patients.csv")
        df.to_csv(csv_path, index=False)

        outbreak_pct = df['outbreak_risk'].mean() * 100
        print(
            f"  {hosp['flag']} {hosp['name']:.<40s} "
            f"{len(df):>5,} patients | "
            f"Outbreak rate: {outbreak_pct:5.1f}%"
        )

        all_data[hosp_id] = df

    total = sum(len(df) for df in all_data.values())
    print(f"\n  Total patients generated: {total:,}")
    print(f"  Data saved to: {config.DATA_DIR}")
    print("=" * 60 + "\n")

    return all_data


def load_hospital_data(hospital_id):
    """
    Load previously generated hospital data from disk.
    
    Args:
        hospital_id: Integer hospital identifier
        
    Returns:
        pandas DataFrame with patient records
    """
    csv_path = os.path.join(
        config.DATA_DIR, f"hospital_{hospital_id}", "patients.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No data found for hospital {hospital_id}. "
            f"Run generate_all_hospital_data() first."
        )
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    generate_all_hospital_data()
