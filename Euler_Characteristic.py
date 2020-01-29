import numpy as np
import euchar.utils
from euchar.surface import images_2D, images_3D, bifiltration
from euchar.filtrations import (alpha_filtration_2D, alpha_filtration_3D,
                                inverse_density_filtration)
from euchar.display import matplotlib_plot, euler_surface_plot
from seaborn import distplot

points_2D = points
simplices_2D, alpha_2D = alpha_filtration_2D(points_2D)
density_2D = inverse_density_filtration(points_2D, simplices_2D, n_neighbors=6)


fig, ax = matplotlib_plot(figsize=(5,3))

_ = distplot(alpha_2D, ax=ax)
ax.set(title="Distribution 2D Alpha param")
_ = distplot(density_2D, ax=ax)
ax.set(title="Density estimate param")
ax.legend()




























