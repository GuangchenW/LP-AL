from .objective_function import BaseObjectiveFunction

import os
import matlab.engine
import numpy as np
from scipy.stats import norm
from ansys.mapdl.core import launch_mapdl

class G_Stiffener2(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="Stiffener", ndim=10, failure_probability=0.002823)
		mapdl = launch_mapdl()
		self.mapdl = mapdl

		mapdl.clear()
		mapdl.prep7()
		print(mapdl)

		mapdl.units("SI")
		self.sfactor = 1e-2 # Sclaing factor from cm to m
		# Material and basic properties (density and Poisson's ratio)
		mapdl.et(1, "PLANE183", kop3=3)
		mapdl.mp("DENS", 1, 2830) # Density in kg/m^3
		mapdl.mp("NUXY", 1, 0.3) # Poisson's ratio
		self.set_properties() # Defalt thickness and Young's modulus

		self.geometry = self.create_geometry()
		self.create_mesh(self.geometry)
		self.set_constraint()
		mapdl.finish()

	def _evaluate(self, x):
		E,d,P1,P2,F1,F2,F3,F4,F5,F6 = x

		response = self.solve(E,d,P1,P2,F1,F2,F3,F4,F5,F6)

		return (3.5e-4-response)*1e4
		#return response

	def variable_definition(self):
		# All variables are normal and have C.O.V. of 0.05, so common generator.
		def get_rn(mean):
			return np.random.normal(mean, mean*0.05)
		E = get_rn(100) # GPa
		d = get_rn(5) # mm
		P1 = get_rn(5000) # Pa
		P2 = get_rn(5000) # Pa
		F1 = get_rn(35239) # N
		F2 = get_rn(23758) # N
		F3 = get_rn(5949) # N
		F4 = get_rn(16245) # N
		F5 = get_rn(10140) # N
		F6 = get_rn(19185) # N

		return [E,d,P1,P2,F1,F2,F3,F4,F5,F6]

	def logpdf(self, x):
		E,d,P1,P2,F1,F2,F3,F4,F5,F6 = self.denormalize_data(x)
		# TODO: Could probably use loop for this with a mean array
		def get_prob(val, mean):
			return norm.logpdf(val, mean, mean*0.05)
		prob = get_prob(E, 100)
		prob += get_prob(d, 5)
		prob += get_prob(P1, 5000)
		prob += get_prob(P2, 5000)
		prob += get_prob(F1, 35239)
		prob += get_prob(F2, 23758)
		prob += get_prob(F3, 5949)
		prob += get_prob(F4, 16245)
		prob += get_prob(F5, 10140)
		prob += get_prob(F6, 19185)

		return prob

############################################################################

	def set_properties(self, thickness=5, E=100):
		mapdl = self.mapdl
		mapdl.r(1, thickness * self.sfactor * 0.1) # Thickness (mm -> m)
		mapdl.mp("EX", 1, E*1e9) # Young's modulus (GPa -> Pa)

	def create_geometry(self):
		mapdl = self.mapdl
		k0 = mapdl.k("", 0, 0, 0)
		k1 = mapdl.k("", 0, 511.552*self.sfactor, 0)
		k2 = mapdl.k("", 337.0659*self.sfactor, 119.594*self.sfactor, 0)
		arc = mapdl.larc(k1, k2, k0, 423.0005*self.sfactor)
		geometry = mapdl.a(k0, k1, k2, arc)

		h1 = mapdl.cyl4(47*self.sfactor, 345.0006*self.sfactor, 20*self.sfactor) # Top hole
		h2 = mapdl.cyl4(58.8659*self.sfactor, 296.1956*self.sfactor, 8.5*self.sfactor) # Second top hole
		h3 = mapdl.cyl4(23.8749*self.sfactor, 200.6504*self.sfactor, 8.5*self.sfactor) # Left most hole
		h4 = mapdl.cyl4(84*self.sfactor, 166.6426*self.sfactor, 33.25*self.sfactor) # Large hole
		h5 = mapdl.cyl4(194.9999*self.sfactor, 229.595*self.sfactor, 8.5*self.sfactor) # Right most hole
		h6 = mapdl.cyl4(167.85*self.sfactor, 121.695*self.sfactor, 8.5*self.sfactor) # Bottom hole

		holes = [h1, h2, h3, h4, h5, h6]

		for h in holes:
			geometry = mapdl.asba(geometry, h)

		#mapdl.lplot(cpos="xy", line_width=3, font_size=26, color_lines=True, background="w")
		return geometry

	def create_mesh(self, geometry):
		mapdl = self.mapdl

		mapdl.lsel("S", "LINE", vmin=1, vmax=3)
		mapdl.lesize("ALL", 10*self.sfactor, kforc=1)
		mapdl.lsel("S", "LINE", vmin=4, vmax=7)
		mapdl.lsel("A", "LINE", vmin=16, vmax=19)
		mapdl.lesize("ALL", ndiv=9, kforc=1)
		mapdl.lsel("S", "LINE", vmin=8, vmax=15)
		mapdl.lsel("A", "LINE", vmin=20, vmax=27)
		mapdl.lesize("ALL", ndiv=6, kforc=1)
		mapdl.lsel("ALL")
		mapdl.smrtsize(trans=3.3)
		mapdl.amesh(self.geometry)

		#mapdl.eplot(vtk=True, cpos="xy", show_edges=True, line_width=2, background="w")

	def set_constraint(self):
		"""
		Apply the fixed displacement constraint to the left edge.
		"""
		mapdl = self.mapdl

		mapdl.lsel("S", "LINE", vmin=2)
		mapdl.nsll("S", 1)
		mapdl.d("ALL", "ALL") # Apply fixed constraint

	def set_boundary_condition(
		self, 
		P1=5000, 
		P2=5000, 
		F1=35239, 
		F2=23758, 
		F3=5949,
		F4=16245,
		F5=10140,
		F6=19185):
		mapdl = self.mapdl

		# Helper function for selecting nodes on lines
		def select_nodes(vmin, vmax=None):
			vmax = vmin if not vmax else vmax
			mapdl.lsel("S", "LINE", vmin=vmin, vmax=vmax)
			mapdl.nsll("S", 1)

		mapdl.sfdele("ALL", "ALL") # Reset pressure loads
		mapdl.fdele("ALL", "ALL") # Reset force loads

		# Apply aerodynamic pressure load
		select_nodes(1)
		mapdl.sf("ALL", "PRES", -P1)

		select_nodes(3)
		mapdl.sf("ALL", "PRES", P2)

		# Apply force load
		select_nodes(16, 19)
		n = mapdl.get(entity="NODE", item1="COUNT")/2
		mapdl.f("ALL", "FY", -F4/n/2)
		mapdl.f("ALL", "FX", -F3/n/2)

		select_nodes(20, 23)
		n = mapdl.get(entity="NODE", item1="COUNT")/2
		mapdl.f("ALL", "FY", F1/n)
		mapdl.f("ALL", "FX", F2/n)

		select_nodes(24, 27)
		n = mapdl.get(entity="NODE", item1="COUNT")/2
		mapdl.f("ALL", "FY", -F5/n)
		mapdl.f("ALL", "FX", F6/n)


	def solve(self,E,d,P1,P2,F1,F2,F3,F4,F5,F6):
		mapdl = self.mapdl
		mapdl.prep7()
		self.set_properties(thickness=d, E=E)
		self.set_boundary_condition(P1,P2,F1,F2,F3,F4,F5,F6)
		mapdl.allsel(mute=True)
		mapdl.solution()
		mapdl.antype("STATIC")
		output = mapdl.solve()
		mapdl.finish()
		result = mapdl.post_processing.nodal_displacement("Y")
		"""
		mapdl.result.plot_nodal_displacement(
			0,
			cpos="xy",
			displacement_factor=1e5*self.sfactor,
			show_displacement=True,
			comp="Y"
		)
		"""
		return abs(result).max()

if __name__ == "__main__":
	test = G_Stiffener2()




