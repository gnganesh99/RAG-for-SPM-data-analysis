import numpy as np
import matplotlib.pyplot as plt

# If stmpy is not installed, install it with: pip install stmpy
import stmpy
import re

# --- Load the data ---
filepath = "./stm_data_code_sample/3D_hyperspectral_data/cits_data.3ds"
smd = stmpy.load(filepath, biasOffset=False)

# --- Helper to find a didv_x (LIX) channel in the grid ---
def find_didv_x_array(smd_obj):
    """
    Return the (nV, Ny, Nx) array for the dI/dV X channel if present.
    Tries common STMPy keys like 'LIX 1 omega (A)'.
    """
    if not hasattr(smd_obj, "grid") or not isinstance(smd_obj.grid, dict):
        return None, None

    priority_keys = [
        "LIX 1 omega (A)",
        "LIX (A)",
        "LI X 1 omega (A)",
        "LIX",
    ]

    for k in priority_keys:
        if k in smd_obj.grid:
            arr = smd_obj.grid[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                return arr, k

    # Heuristic search
    for k in smd_obj.grid.keys():
        kl = k.lower()
        if ("lix" in kl or ("x" in kl and "omega" in kl) or "didv" in kl) and isinstance(smd_obj.grid[k], np.ndarray):
            arr = smd_obj.grid[k]
            if arr.ndim == 3:
                return arr, k

    return None, None

# --- Parse frame size (in meters) from header ---
def get_frame_size_from_header(header_dict):
    """
    Extract frame size (Lx, Ly) in meters from header['Grid settings'] using regex,
    following the approach shown in the retrieved context.
    """
    d_line = header_dict.get("Grid settings", "")
    # Match numbers including scientific notation
    match_number = re.compile(r'-?\s*[0-9]+\.?[0-9]*(?:[Ee]\s*-?\s*[0-9]+)?')
    nums = [float(x.replace(" ", "")) for x in re.findall(match_number, d_line)]
    # Context indicates indices 2 and 3 correspond to frame size
    if len(nums) >= 4:
        return float(nums[2]), float(nums[3])
    # Fallback: try Grid dim if available (not physical size, but avoids crash)
    raise ValueError("Could not parse frame size from header['Grid settings'].")

# --- Determine nearest pixel to the probe point (meters) ---
probe_point = (2e-7, 2e-7)  # (x, y) in meters; 2e-7 m == 200 nm

# Get bias axis and data shapes
V = np.asarray(smd.en)                  # shape (nV,)
nV, Ny, Nx = smd.I.shape               # current array shape

# Frame size in meters
Lx, Ly = get_frame_size_from_header(smd.header)

# Build coordinate vectors spanning the frame (assumed [0, Lx] and [0, Ly])
x_vec = np.linspace(0, Lx, Nx)
y_vec = np.linspace(0, Ly, Ny)

# Find nearest indices
px, py = probe_point
ix = int(np.argmin(np.abs(x_vec - px)))
iy = int(np.argmin(np.abs(y_vec - py)))

x_actual = x_vec[ix]
y_actual = y_vec[iy]

# --- Locate dI/dV (X) data ---
didv_arr, didv_key = find_didv_x_array(smd)
if didv_arr is None:
    print("Could not locate a dI/dV X (LIX) channel in this .3ds file.")
    print("Available grid keys:")
    print(list(smd.grid.keys()) if hasattr(smd, "grid") else "No grid present.")
else:
    # Extract spectrum at the nearest point; didv_arr has shape (nV, Ny, Nx)
    didv_spectrum = didv_arr[:, iy, ix]

    # --- Plot dI/dV (X) vs V at the selected point ---
    plt.figure(figsize=(6, 4))
    plt.plot(V, didv_spectrum, label=f"{didv_key}")
    plt.xlabel("Bias (V)")
    plt.ylabel("dI/dV (A)")
    plt.title(f"dI/dV (X) vs V at ({x_actual:.2e} m, {y_actual:.2e} m) [pix ({ix},{iy})]")
    plt.legend()
    plt.tight_layout()
    plt.show()
