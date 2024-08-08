import numpy as np

from ansys.mapdl.core import launch_mapdl

class G_Stiffener2():
	def __init__(self):
		mapdl = launch_mapdl()
		self.mapdl = mapdl

		mapdl.clear()
		mapdl.prep7()
		print(mapdl)

		mapdl.units("SI")
		mapdl.et(1, "PLANE183", kop3=3)

		self.set_properties()
		print(mapdl.mplist())
		self.geometry = self.create_geometry()
		self.create_mesh(self.geometry)

	def set_properties(self, thickness=0.5, E=100e9):
		mapdl = self.mapdl
		mapdl.r(1, thickness) # Thickness
		mapdl.mp("EX", 1, E) # Young's modulus
		mapdl.mp("DENS", 1, 2830) # Density in kg/m^3
		mapdl.mp("NUXY", 1, 0.3) # Poisson's ratio

	def create_geometry(self):
		mapdl = self.mapdl
		k0 = mapdl.k("", 0, 0, 0)
		k1 = mapdl.k("", 0, 511.552, 0)
		k2 = mapdl.k("", 337.0659, 119.594, 0)
		arc = mapdl.larc(k1, k2, k0, 423.0005)
		geometry = mapdl.a(k0, k1, k2, arc)

		h1 = mapdl.cyl4(47, 345.0006, 20) # Top hole
		h2 = mapdl.cyl4(58.8659, 296.1956, 8.5) # Second top hole
		h3 = mapdl.cyl4(23.8749, 200.6504, 8.5) # Left most hole
		h4 = mapdl.cyl4(84, 166.6426, 33.25) # Large hole
		h5 = mapdl.cyl4(194.9999, 229.595, 8.5) # Right most hole
		h6 = mapdl.cyl4(167.85, 121.695, 8.5) # Bottom hole

		holes = [h1, h2, h3, h4, h5, h6]

		for h in holes:
			geometry = mapdl.asba(geometry, h)

		return geometry

	def create_mesh(self, geometry):
		mapdl = self.mapdl
		mapdl.lsel("ALL")
		mapdl.esize(10)
		mapdl.amesh(self.geometry)
		mapdl.eplot(
			vtk=True,
			cpos="xy",
			show_edges=True,
			show_axes=False,
			line_width=2,
			background="w",
		)


if __name__ == "__main__":
	test = G_Stiffener2()




