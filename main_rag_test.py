import os
import numpy as np
import matplotlib.pyplot as plt

import stmpy

def get_topography(smd):
    # Try common attribute names
    for name in ['topo', 'topography', 'z', 'Z', 'Topo', 'Topography', 'height', 'Height']:
        if hasattr(smd, name):
            topo = getattr(smd, name)
            topo = np.asarray(topo)
            if topo.ndim == 2:
                return topo
            if topo.ndim >= 3:
                return topo[0]
    # Try within dict-like attributes
    for obj in [getattr(smd, 'signals', None), getattr(smd, 'grid', None), getattr(smd, '__dict__', None)]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if any(t in lk for t in ['topo', 'height', 'topography', ' z', 'z (', '(z)']):
                    arr = np.asarray(v)
                    if arr.ndim == 2:
                        return arr
                    if arr.ndim >= 3:
                        return arr[0]
    return None

def nearest_index(value, array):
    array = np.asarray(array, dtype=float).ravel()
    return int(np.argmin(np.abs(array - float(value))))

def main():
    filepath = './stm_data_code_sample/3D_hyperspectral_data/cits_data.3ds'
    smd = stmpy.load(filepath, biasOffset=False)

    # Extract topography (Z/height)
    topo = get_topography(smd)

    # Current map at probe bias 1.2 V
    V = np.asarray(smd.en, dtype=float).ravel()
    probe_bias = 1.2
    iv = nearest_index(probe_bias, V)
    V_actual = V[iv]

    Icube = np.asarray(smd.I)
    # Expect shape (nV, nx, ny); if not, try to reconcile basic alternatives
    if Icube.ndim != 3:
        raise ValueError("Unexpected current data dimensionality.")
    if Icube.shape[0] == V.size:
        current_map = Icube[iv]
    elif Icube.shape[-1] == V.size:
        current_map = np.moveaxis(Icube, -1, 0)[iv]
    else:
        # Fallback: assume first axis corresponds to energy
        current_map = Icube[iv]

    # Plot
    if topo is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(topo, origin='lower', aspect='auto')
        axes[0].set_title('Topography (Z/height)')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(current_map, origin='lower', aspect='auto')
        axes[1].set_title(f'Current map at {V_actual:.3f} V')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
    else:
        plt.figure(figsize=(5, 4))
        im1 = plt.imshow(current_map, origin='lower', aspect='auto')
        plt.title(f'Current map at {V_actual:.3f} V')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
