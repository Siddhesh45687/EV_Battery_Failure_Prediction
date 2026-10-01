import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="EV Battery AI",
    page_icon="🔋",
    layout="wide"
)

model = joblib.load("model/xgb_model (1).pkl")

# ---------------- MAPS ----------------
M = {
    "brand": dict(zip(
        ["Audi","BMW","BYD","Ford","GM","Honda","Hyundai","Kia","Lucid","MG",
         "Mahindra","Mercedes","Nissan","Renault","Rivian","Tata","Tesla",
         "Toyota","Volkswagen","Volvo"], range(20)
    )),
    "type": dict(zip(
        ["Crossover","Hatchback","SUV","Sedan","Truck","Van"], range(6)
    )),
    "man": dict(zip(
        ["BYD Battery","CATL","Envision AESC","Guoxuan",
         "LG Energy Solution","Northvolt","Panasonic",
         "SK On","Samsung SDI"], range(9)
    )),
    "chem": dict(zip(
        ["LFP","LMO","LTO","NCA","NMC"], range(5)
    )),
    "drive": dict(zip(
        ["AWD","FWD","RWD"], range(3)
    )),
    "terrain": dict(zip(
        ["Coastal","Desert","Flat","Hilly","Mountainous"], range(5)
    )),
    "usage": {
        "Fleet": 0,
        "Private": 1
    }
}


# =========================================================
# BLACK DASHBOARD UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 20%, #172554 0%, transparent 30%),
        radial-gradient(circle at 85% 80%, #083344 0%, transparent 30%),
        #05070b;
    color: #f8fafc;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Header */
.hero {
    padding: 28px 32px;
    border-radius: 20px;
    margin-bottom: 25px;

    background:
        linear-gradient(135deg, #0f172a, #111827);

    border: 1px solid #1e293b;

    box-shadow:
        0 0 35px rgba(37,99,235,0.20),
        inset 0 0 30px rgba(255,255,255,0.02);
}

.hero h1 {
    margin: 0;
    color: #38bdf8;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    margin: 8px 0 0 0;
    color: #94a3b8;
    font-size: 16px;
}

/* Section titles */
h2, h3 {
    color: #e2e8f0 !important;
}

/* Cards */
div[data-testid="stVerticalBlock"] {
    color: #e2e8f0;
}

/* Input labels */
label {
    color: #cbd5e1 !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Slider */
div[data-testid="stSlider"] {
    padding: 5px 0;
}

/* Slider track */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #38bdf8 !important;
}

/* Number/value text */
div[data-testid="stSlider"] p {
    color: #e2e8f0 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 16px;
    font-weight: 700;

    box-shadow:
        0 0 20px rgba(6,182,212,0.25);

    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 0 30px rgba(6,182,212,0.45);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #1e293b;
    padding: 18px;
    border-radius: 15px;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.35);
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div {
    background-color: #1e293b;
    border-radius: 10px;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
}

/* Success / error boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #1e293b !important;
}

/* Expander */
details {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}

</style>

<div class="hero">
    <h1>🔋 EV Battery AI</h1>
    <p>Machine Learning Battery Failure Risk Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT HELPER ----------------

def S(label, key):
    return st.selectbox(label, M[key])


# =========================================================
# VEHICLE
# =========================================================

st.subheader("🚗 Vehicle")

c1, c2, c3 = st.columns(3)

with c1:
    brand = S("Brand", "brand")
    vtype = S("Vehicle Type", "type")
    year = st.slider("Year", 2015, 2024, 2020)

with c2:
    man = S("Battery Manufacturer", "man")
    chem = S("Chemistry", "chem")
    drive = S("Drive", "drive")

with c3:
    usage = S("Usage", "usage")
    terrain = S("Terrain", "terrain")
    cap = st.slider("Capacity (kWh)", 25., 150., 75.)


# =========================================================
# BATTERY
# =========================================================

st.subheader("🔋 Battery")

c1, c2, c3 = st.columns(3)

with c1:
    health = st.slider("Health %", 40, 100, 90)
    soh = st.slider("SOH %", 40, 100, 90)
    cycles = st.slider("Cycles", 0, 1500, 250)

with c2:
    soc = st.slider("SOC %", 0, 100, 70)
    dod = st.slider("DOD %", 0, 100, 50)
    resistance = st.slider("Resistance", .01, .30, .05)

with c3:
    temp = st.slider("Avg Temp °C", -10, 70, 30)
    maxt = st.slider("Max Temp °C", 0, 100, 40)
    voltage = st.slider("Voltage Imbalance", 0., .20, .02)


# =========================================================
# USAGE
# =========================================================

st.subheader("⚡ Usage")

c1, c2, c3 = st.columns(3)

with c1:
    fast = st.slider("Fast Charge", 0., 1., .30)
    power = st.slider("Charge Power", 1., 100., 20.)
    over = st.slider("Overcharge", 0, 30, 0)

with c2:
    interrupt = st.slider("Interruptions", 0, 30, 2)
    accel = st.slider("Acceleration", 0, 100, 20)
    brake = st.slider("Hard Braking", 0, 100, 20)

with c3:
    bms = st.slider("BMS Warnings", 0, 30, 0)
    faults = st.slider("Previous Faults", 0, 20, 0)
    sensor = st.slider("Sensor Faults", 0, 20, 0)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍  PREDICT FAILURE",
    type="primary",
    use_container_width=True
):

    age = 2026 - year

    stress = min(
        100,
        (100-health)*.3 +
        (100-soh)*.2 +
        cycles/15 +
        max(0, temp-35)*1.5 +
        resistance*100 +
        voltage*100 +
        over*2 +
        bms*2 +
        faults*3
    )

    X = {
        "vehicle_brand": M["brand"][brand],
        "vehicle_type": M["type"][vtype],
        "manufacturing_year": year,
        "battery_manufacturer": M["man"][man],
        "battery_chemistry": M["chem"][chem],
        "battery_capacity_kwh": cap,
        "drive_type": M["drive"][drive],
        "odometer_km": cycles*350,
        "vehicle_age_years": age,
        "fleet_or_private": M["usage"][usage],

        "cycle_count": cycles,
        "battery_health_percent": health,
        "state_of_charge": soc,
        "depth_of_discharge": dod,
        "state_of_health": soh,

        "cell_voltage_avg": 3.7,
        "cell_voltage_std": voltage/10,
        "pack_voltage": 350,

        "cell_temperature_avg": temp,
        "cell_temperature_max": maxt,
        "internal_resistance": resistance,

        "charge_efficiency": max(60, 95-fast*15),
        "discharge_efficiency": 90,

        "remaining_capacity": cap*soh/100,
        "capacity_loss_percent": 100-soh,

        "charging_cycles_last_month": cycles/12,
        "fast_charge_ratio": fast,
        "slow_charge_ratio": 1-fast,

        "average_charge_power_kw": power,
        "average_charging_time": 3,
        "overnight_charging_ratio": .4,
        "home_charging_ratio": .7,

        "charging_interruptions": interrupt,
        "overcharge_events": over,

        "average_speed": 50,
        "average_trip_distance": 30,

        "aggressive_acceleration_score": accel,
        "hard_braking_score": brake,
        "regenerative_braking_usage": 50,

        "highway_driving_ratio": .4,
        "city_driving_ratio": .6,

        "daily_distance": 40,
        "average_ambient_temperature": 25,
        "maximum_temperature": maxt,
        "minimum_temperature": 15,

        "humidity": 60,
        "altitude": 100,

        "terrain_type": M["terrain"][terrain],

        "dust_exposure": 20,
        "last_service_days": 100,

        "cooling_system_health":
            max(30, 100-(maxt-40)*2),

        "firmware_updates": 3,

        "previous_faults": faults,

        "maintenance_score":
            max(20, 100-faults*5),

        "thermal_runaway_risk":
            stress/2,

        "voltage_imbalance": voltage,

        "temperature_variance":
            max(1, (maxt-temp)/3),

        "sensor_fault_count": sensor,
        "BMS_warning_count": bms,

        "abnormal_voltage_events":
            int(voltage*20),

        "battery_stress_index":
            stress,

        "aging_score":
            min(100, (100-health)+age*2),

        "thermal_health_score":
            max(0, 100-(maxt-40)*2),

        "charging_quality_score":
            max(0, 100-fast*40-over*2),

        "driving_stress_score":
            (accel+brake)/2,

        "predicted_remaining_life_cycles":
            max(50, 1500-cycles)
    }

    X = pd.DataFrame([X])

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_)

    p = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    a, b, c = st.columns(3)

    a.metric(
        "Failure Probability",
        f"{p:.2%}"
    )

    b.metric(
        "Battery SOH",
        f"{soh}%"
    )

    c.metric(
        "Battery Stress",
        f"{stress:.0f}/100"
    )

    st.progress(float(p))

    if pred:
        st.error(
            f"⚠️ FAILURE RISK DETECTED — Probability: {p:.2%}"
        )
    else:
        st.success(
            f"✅ BATTERY OPERATING NORMALLY — Probability: {p:.2%}"
        )