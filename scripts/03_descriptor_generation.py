from pathlib import Path
import re

import numpy as np
import pandas as pd
from mendeleev import element


METALS = ("Cu", "Co", "Fe", "Ni", "Pd", "Ru", "Ag", "Mn", "Zn", "Pt", "Ir")

MEAN_DESCRIPTORS = (
    "Atomic Number",
    "Period",
    "Group",
    "Atomic Weight",
    "Atomic Radius",
    "Metallic Radius",
    "EN Pauling",
    "Ionization Energy1st",
    "nValence",
    "s Valence",
    "d Valence",
    "Melting Point",
    "Density",
    "Thermal Conductivity",
)

VAR_DESCRIPTORS = {
    "Atomic Radius Var": "Atomic Radius",
    "Metallic Radius Var": "Metallic Radius",
    "EN Var": "EN Pauling",
    "Ionization Energy Var": "Ionization Energy1st",
    "nValence Var": "nValence",
    "s Valence Var": "s Valence",
    "d Valence Var": "d Valence",
}

THERMAL_CONDUCTIVITY_FIX = {
    "Mn": 7.8, # Missing in mendeleev 1.1.0; value used to reproduce the reference descriptor table.
}


def format_number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def material_name(row):
    name = f"Cu{format_number(row['Cu'])}"

    for metal in METALS[1:]:
        if row[metal] > 0:
            name += f"{metal}{format_number(row[metal])}"

    return name


def valence_counts(el):
    matches = re.findall(r"(\d+[spd])(\d+)", str(el.ec))
    max_n = max(int(orbital[0]) for orbital, _ in matches)

    s_valence = 0
    d_valence = 0

    for orbital, count in matches:
        n = int(orbital[0])
        shell = orbital[1]
        count = int(count)

        if n == max_n and shell == "s":
            s_valence += count
        elif n == max_n - 1 and shell == "d":
            d_valence += count

    return s_valence + d_valence, s_valence, d_valence


def element_descriptor_table():
    rows = {}

    for metal in METALS:
        el = element(metal)
        n_valence, s_valence, d_valence = valence_counts(el)

        thermal_conductivity = el.thermal_conductivity
        if thermal_conductivity is None:
            thermal_conductivity = THERMAL_CONDUCTIVITY_FIX[metal]

        rows[metal] = {
            "Atomic Number": el.atomic_number,
            "Period": el.period,
            "Group": el.group_id,
            "Atomic Weight": float(el.atomic_weight),
            "Atomic Radius": el.atomic_radius,
            "Metallic Radius": el.metallic_radius_c12,
            "EN Pauling": el.en_pauling,
            "Ionization Energy1st": el.ionenergies[1],
            "nValence": n_valence,
            "s Valence": s_valence,
            "d Valence": d_valence,
            "Melting Point": el.melting_point,
            "Density": el.density,
            "Thermal Conductivity": thermal_conductivity,
        }

    return pd.DataFrame(rows).T


def molar_weights(compositions, atomic_weights):
    moles = compositions / atomic_weights
    return moles / moles.sum(axis=1, keepdims=True)


def generate_descriptors(compositions, element_descriptors):
    atomic_weights = element_descriptors["Atomic Weight"].to_numpy(dtype=float)
    weights = molar_weights(compositions, atomic_weights)

    mean_matrix = element_descriptors.loc[:, MEAN_DESCRIPTORS].to_numpy(dtype=float)
    descriptor_data = weights @ mean_matrix

    descriptors = pd.DataFrame(descriptor_data, columns=MEAN_DESCRIPTORS)

    for output_name, source_name in VAR_DESCRIPTORS.items():
        values = element_descriptors[source_name].to_numpy(dtype=float)
        mean_values = weights @ values
        descriptors[output_name] = np.sum(
            weights * (values - mean_values[:, None]) ** 2,
            axis=1,
        )

    return descriptors


def main():
    repo_root = Path(__file__).resolve().parents[1]

    input_path = repo_root / "data" / "candidate_space_13650.xlsx"
    output_path = repo_root / "data" / "candidate_space_descriptors_13650.xlsx"

    candidate_space = pd.read_excel(input_path, engine="openpyxl")
    compositions = candidate_space.loc[:, METALS].astype(float)

    element_descriptors = element_descriptor_table()
    descriptors = generate_descriptors(
        compositions.to_numpy(dtype=float),
        element_descriptors,
    )

    result = pd.concat(
        [
            compositions.apply(material_name, axis=1).rename("Materials"),
            compositions,
            descriptors,
        ],
        axis=1,
    )

    result.to_excel(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()