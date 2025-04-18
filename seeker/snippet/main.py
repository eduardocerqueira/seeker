#date: 2025-04-18T17:05:42Z
#url: https://api.github.com/gists/bfce68ec73277f738bf29ad62c2e8958
#owner: https://api.github.com/users/G1r00t

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle, Polygon, Circle, Arrow
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

class PVModuleParameters:
    def __init__(self):

        self.T_ref = 298.15  
        self.I_sc_ref = 9.35  
        self.V_oc_ref = 37.5  
        self.I_mp_ref = 8.75  
        self.V_mp_ref = 31.4  
        self.P_max_ref = 275  

        self.length = 1.65  
        self.width = 0.992  
        self.A = self.length * self.width  
        self.n_cells = 60  
        self.cell_area = 0.0256  
        self.weight = 18.6  
        self.glass_thickness = 0.0032  
        self.EVA_thickness = 0.0005  
        self.backsheet_thickness = 0.0003  

        self.G_ref = 1000  
        self.sigma = 5.67e-8  
        self.emissivity_front = 0.91  
        self.emissivity_back = 0.85  
        self.absorptivity = 0.95  
        self.k = 1.38e-23  
        self.q = 1.602e-19  
        self.n_diode = 1.2  
        self.FF_ref = self.P_max_ref / (self.I_sc_ref * self.V_oc_ref)  
        self.Rs = 0.35  
        self.Rsh = 350  

        self.beta_V = -0.0031  
        self.alpha_I = 0.0005  
        self.gamma_P = -0.004  

        self.NOCT = 318.15  
        self.T_ambient = 298.15  
        self.wind_speed = 1.0  

        self.cell_type = "mono-Si"  
        self.spectral_response = {  
            300: 0.0, 350: 0.1, 400: 0.4, 450: 0.6, 
            500: 0.75, 550: 0.85, 600: 0.93, 650: 0.99, 
            700: 1.0, 750: 0.95, 800: 0.8, 850: 0.65, 
            900: 0.4, 950: 0.2, 1000: 0.1, 1050: 0.05, 1100: 0.0
        }

        self.CTM_optical = 0.97  
        self.CTM_resistive = 0.98  
        self.CTM_mismatch = 0.985  

        self.age = 0  
        self.annual_degradation = 0.005  

        self.tilt = 30  
        self.azimuth = 180  

params = PVModuleParameters()

class SolarPanelModel:
    def __init__(self, parameters):
        self.params = parameters

    def calculate_cell_temperature(self, T_amb, G, wind_speed=None, tilt=None):
        """
        Calculate cell temperature using improved model accounting for mounting configuration and wind

        Based on Faiman model: T_cell = T_amb + G / (U0 + U1 * wind_speed)
        """
        if wind_speed is None:
            wind_speed = self.params.wind_speed

        if tilt is None:
            tilt = self.params.tilt

        U0 = 25.0  
        U1 = 6.84  

        tilt_rad = np.radians(tilt)
        tilt_factor = 1.0 - 0.1 * np.sin(tilt_rad)  

        delta_T = G / (U0 * tilt_factor + U1 * wind_speed)
        T_cell = T_amb + delta_T

        return T_cell

    def spectral_correction_factor(self, T_amb, AM=1.5):
        """
        Calculate spectral correction factor based on Air Mass and ambient temperature

        This is a simplified model based on empirical data
        Air Mass (AM) is the optical path length through Earth's atmosphere
        """

        AM_correction = 1.0 - 0.03 * (AM - 1.5)
        T_correction = 1.0 - 0.002 * (T_amb - self.params.T_ref)

        return AM_correction * T_correction

    def calculate_incidence_angle_modifier(self, angle_of_incidence):
        """
        Calculate the incidence angle modifier that reduces effective irradiance
        due to increased reflection at higher incidence angles
        """

        if angle_of_incidence < 50:
            IAM = 1.0 - 1.098e-4 * angle_of_incidence - 6.267e-6 * angle_of_incidence**2
        else:
            IAM = 1.0 - 2.5836e-3 * (angle_of_incidence - 50)

        return max(0, IAM)

    def calculate_effective_irradiance(self, G, angle_of_incidence=0, soiling_factor=0.98):
        """
        Calculate effective irradiance accounting for incidence angle and soiling
        """
        IAM = self.calculate_incidence_angle_modifier(angle_of_incidence)
        G_effective = G * IAM * soiling_factor
        return G_effective

    def calculate_diode_parameters(self, T_cell):
        """
        Calculate parameters for the single-diode model as a function of temperature
        """

        V_t = self.params.k * T_cell / self.params.q

        E_g = 1.12 * (1 - 0.0002677 * (T_cell - self.params.T_ref))  
        I_0_factor = ((T_cell / self.params.T_ref) ** 3) * np.exp(-E_g * self.params.q / (self.params.k * T_cell))
        I_0 = 1e-9 * I_0_factor  

        return V_t, I_0

    def calculate_IV_parameters(self, T_cell, G_effective):
        """
        Calculate I-V curve parameters adjusted for operating conditions
        """

        V_oc = self.params.V_oc_ref + self.params.beta_V * (T_cell - self.params.T_ref)
        I_sc = self.params.I_sc_ref * (G_effective / self.params.G_ref) * \
               (1 + self.params.alpha_I * (T_cell - self.params.T_ref))

        V_t, I_0 = self.calculate_diode_parameters(T_cell)

        v_oc_norm = V_oc / (self.params.n_diode * V_t)  
        FF_0 = (v_oc_norm - np.log(v_oc_norm + 0.72)) / (v_oc_norm + 1)  

        rs = self.params.Rs / (V_oc / I_sc)  
        FF = FF_0 * (1 - 1.1 * rs) * (1 - (V_t / V_oc) * np.log(1 + (V_oc / (self.params.Rsh * I_sc))))

        FF_module = FF * self.params.CTM_resistive * self.params.CTM_mismatch

        return V_oc, I_sc, FF_module

    def calculate_thermal_losses(self, T_cell, T_amb):
        """
        Calculate thermal losses through radiation, convection, and conduction
        """

        Q_rad_front = self.params.sigma * self.params.emissivity_front * \
                      self.params.A * (T_cell**4 - T_amb**4)
        Q_rad_back = self.params.sigma * self.params.emissivity_back * \
                     self.params.A * (T_cell**4 - T_amb**4)

        h_conv = 5 + 3.8 * self.params.wind_speed  
        Q_conv = h_conv * self.params.A * (T_cell - T_amb)

        Q_thermal = Q_rad_front + Q_rad_back + Q_conv

        return Q_thermal

    def aging_derating_factor(self):
        """
        Calculate derating factor due to panel aging
        """
        return 1.0 - self.params.annual_degradation * self.params.age

    def solar_panel_performance(self, T_amb, G, wind_speed=None, tilt=None, 
                                angle_of_incidence=0, soiling_factor=0.98, 
                                spectral_factor=None, AM=1.5):
        """
        Calculate solar panel performance accounting for multiple factors
        """

        T_cell = self.calculate_cell_temperature(T_amb, G, wind_speed, tilt)

        G_effective = self.calculate_effective_irradiance(G, angle_of_incidence, soiling_factor)

        if spectral_factor is None:
            spectral_factor = self.spectral_correction_factor(T_amb, AM)

        G_effective = G_effective * spectral_factor

        V_oc, I_sc, FF = self.calculate_IV_parameters(T_cell, G_effective)

        P_electrical = FF * V_oc * I_sc

        P_electrical = P_electrical * self.params.CTM_optical

        Q_thermal = self.calculate_thermal_losses(T_cell, T_amb)

        P_electrical = P_electrical * self.aging_derating_factor()

        P_net = P_electrical - Q_thermal

        efficiency = max(0, P_electrical / (G * self.params.A) * 100)
        efficiency_net = max(0, P_net / (G * self.params.A) * 100)

        return {
            'T_cell': T_cell,
            'G_effective': G_effective,
            'V_oc': V_oc,
            'I_sc': I_sc,
            'FF': FF,
            'P_electrical': P_electrical,
            'Q_thermal': Q_thermal,
            'P_net': P_net,
            'efficiency': efficiency,
            'efficiency_net': efficiency_net
        }

    def calculate_IV_curve(self, T_cell, G_effective, V_points=100):
        """
        Calculate the full I-V curve for given conditions
        """

        V_oc, I_sc, _ = self.calculate_IV_parameters(T_cell, G_effective)
        V_t, I_0 = self.calculate_diode_parameters(T_cell)

        V = np.linspace(0, V_oc, V_points)

        I = np.zeros_like(V)

        for i, v in enumerate(V):

            I_guess = I_sc - I_0 * (np.exp(v / (self.params.n_diode * V_t)) - 1)

            max_iter = 10
            tolerance = 1e-6
            for _ in range(max_iter):

                f = I_guess - I_sc + I_0 * (np.exp((v + I_guess * self.params.Rs) / 
                                                  (self.params.n_diode * V_t)) - 1) + \
                    (v + I_guess * self.params.Rs) / self.params.Rsh

                df = 1 + I_0 * np.exp((v + I_guess * self.params.Rs) / 
                                     (self.params.n_diode * V_t)) * \
                    self.params.Rs / (self.params.n_diode * V_t) + self.params.Rs / self.params.Rsh

                delta = f / df
                I_guess = I_guess - delta

                if abs(delta) < tolerance:
                    break

            I[i] = max(0, I_guess)  

        P = V * I

        return V, I, P

model = SolarPanelModel(params)

ambient_temps = np.linspace(273.15, 323.15, 50)  
irradiance_levels = [400, 600, 800, 1000, 1200]  
wind_speeds = [0.5, 1.0, 3.0]  
tilt_angles = [0, 15, 30, 45]  

results = {}
for G in irradiance_levels:
    results[G] = {
        'T_cell': [],
        'efficiency': [],
        'efficiency_net': [],
        'P_electrical': [],
        'P_net': [],
        'V_oc': [],
        'I_sc': [],
        'FF': []
    }

    for T in ambient_temps:
        performance = model.solar_panel_performance(T, G)
        for key in results[G]:
            results[G][key].append(performance[key])

def create_plots():

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    ax1 = axes[0, 0]
    for G in irradiance_levels:
        ax1.plot(ambient_temps - 273.15, results[G]['efficiency'], 
                 label=f"G={G} W/m²", linewidth=2)

    ax1.set_title("Solar Panel Efficiency vs. Ambient Temperature", fontsize=14)
    ax1.set_xlabel("Ambient Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Efficiency (%)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[0, 1]
    for G in irradiance_levels:
        ax2.plot(ambient_temps - 273.15, results[G]['P_electrical'], 
                 label=f"G={G} W/m²", linewidth=2)

    ax2.set_title("Electrical Power Output vs. Ambient Temperature", fontsize=14)
    ax2.set_xlabel("Ambient Temperature (°C)", fontsize=12)
    ax2.set_ylabel("Power Output (W)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for G in irradiance_levels:
        ax3.plot(ambient_temps - 273.15, np.array(results[G]['T_cell']) - 273.15, 
                 label=f"G={G} W/m²", linewidth=2)

    ax3.set_title("Cell Temperature vs. Ambient Temperature", fontsize=14)
    ax3.set_xlabel("Ambient Temperature (°C)", fontsize=12)
    ax3.set_ylabel("Cell Temperature (°C)", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for G in irradiance_levels:
        ax4.plot(ambient_temps - 273.15, results[G]['FF'], 
                 label=f"G={G} W/m²", linewidth=2)

    ax4.set_title("Fill Factor vs. Ambient Temperature", fontsize=14)
    ax4.set_xlabel("Ambient Temperature (°C)", fontsize=12)
    ax4.set_ylabel("Fill Factor", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("solar_panel_performance.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_schematic():
    fig, ax = plt.subplots(figsize=(12, 8))

    panel_color = '#ADD8E6'  
    cell_color = '#1E3F66'   
    frame_color = '#A9A9A9'  
    sun_color = '#FFD700'    
    arrow_color = '#FF6347'  
    glass_color = '#E6F2FF'  
    backsheet_color = '#FFFFFF'  
    EVA_color = '#F0F0F0'    

    ax.set_facecolor('#F5F5F5')  

    panel_width = 2
    panel_height = 3
    panel_thickness = 0.1

    tilt_angle = np.radians(30)

    def rotate(x, y, angle):
        return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

    corners = [
        (-panel_width/2, -panel_height/2),  
        (panel_width/2, -panel_height/2),   
        (panel_width/2, panel_height/2),    
        (-panel_width/2, panel_height/2)    
    ]

    rotated_corners = [rotate(x, y, tilt_angle) for x, y in corners]

    panel = Polygon(rotated_corners, closed=True, edgecolor=frame_color, facecolor=panel_color, 
                    linewidth=2, alpha=0.8)
    ax.add_patch(panel)

    n_rows, n_cols = 5, 3
    cell_width = panel_width * 0.9 / n_cols
    cell_height = panel_height * 0.9 / n_rows
    cell_margin = 0.05

    for i in range(n_rows):
        for j in range(n_cols):

            cell_x = -panel_width/2 + panel_width*0.05 + j * (cell_width + cell_margin)
            cell_y = -panel_height/2 + panel_height*0.05 + i * (cell_height + cell_margin)

            rot_x, rot_y = rotate(cell_x, cell_y, tilt_angle)

            cell = Rectangle((rot_x, rot_y), cell_width, cell_height, 
                           angle=np.degrees(tilt_angle), 
                           edgecolor='black', facecolor=cell_color, linewidth=0.5)
            ax.add_patch(cell)

    sun_x, sun_y = 3, 3
    sun_radius = 0.5
    sun = Circle((sun_x, sun_y), sun_radius, color=sun_color, alpha=0.8)
    ax.add_patch(sun)

    n_rays = 12
    ray_length = 0.3
    for i in range(n_rays):
        angle = 2 * np.pi * i / n_rays
        end_x = sun_x + (sun_radius + ray_length) * np.cos(angle)
        end_y = sun_y + (sun_radius + ray_length) * np.sin(angle)
        ray = plt.Line2D([sun_x + sun_radius * np.cos(angle), end_x], 
                         [sun_y + sun_radius * np.sin(angle), end_y], 
                         color=sun_color, linewidth=1.5)
        ax.add_line(ray)

    n_arrows = 5
    arrow_start_x = sun_x - 0.8
    arrow_start_y = sun_y - 0.8
    arrow_length = 3
    arrow_angle = np.radians(-40)  

    for i in range(n_arrows):
        offset = i * 0.4 - (n_arrows-1) * 0.2
        start_x = arrow_start_x + offset
        start_y = arrow_start_y + offset
        dx = arrow_length * np.cos(arrow_angle)
        dy = arrow_length * np.sin(arrow_angle)

        plt.arrow(start_x, start_y, dx, dy, head_width=0.15, head_length=0.2,
                  fc=arrow_color, ec=arrow_color, alpha=0.7)

    detail_x, detail_y = -3, 0
    detail_width = 2
    detail_height = 1.5

    glass_thickness = 0.15
    eva_top_thickness = 0.05
    cell_thickness = 0.05
    eva_bottom_thickness = 0.05
    backsheet_thickness = 0.03

    detail_bg = Rectangle((detail_x - 0.2, detail_y - 0.3), 
                        detail_width + 0.4, detail_height + 0.6,
                        facecolor='white', edgecolor='black', alpha=0.7)
    ax.add_patch(detail_bg)

    glass = Rectangle((detail_x, detail_y + eva_top_thickness + cell_thickness + eva_bottom_thickness + backsheet_thickness), 
                     detail_width, glass_thickness, 
                     edgecolor='black', facecolor=glass_color)
    ax.add_patch(glass)

    eva_top = Rectangle((detail_x, detail_y + cell_thickness + eva_bottom_thickness + backsheet_thickness), 
                      detail_width, eva_top_thickness, 
                      edgecolor='black', facecolor=EVA_color)
    ax.add_patch(eva_top)

    cell_layer = Rectangle((detail_x, detail_y + eva_bottom_thickness + backsheet_thickness), 
                         detail_width, cell_thickness, 
                         edgecolor='black', facecolor=cell_color)
    ax.add_patch(cell_layer)

    eva_bottom = Rectangle((detail_x, detail_y + backsheet_thickness), 
                         detail_width, eva_bottom_thickness, 
                         edgecolor='black', facecolor=EVA_color)
    ax.add_patch(eva_bottom)

    backsheet = Rectangle((detail_x, detail_y), 
                        detail_width, backsheet_thickness, 
                        edgecolor='black', facecolor=backsheet_color)
    ax.add_patch(backsheet)

    plt.text(detail_x + detail_width + 0.1, detail_y + backsheet_thickness/2, 'Backsheet', 
             va='center', ha='left', fontsize=9)
    plt.text(detail_x + detail_width + 0.1, detail_y + backsheet_thickness + eva_bottom_thickness/2, 
             'EVA Encapsulant', va='center', ha='left', fontsize=9)
    plt.text(detail_x + detail_width + 0.1, detail_y + backsheet_thickness + eva_bottom_thickness + cell_thickness/2, 
             'Solar Cell', va='center', ha='left', fontsize=9)
    plt.text(detail_x + detail_width + 0.1, detail_y + backsheet_thickness + eva_bottom_thickness + cell_thickness + eva_top_thickness/2, 
             'EVA Encapsulant', va='center', ha='left', fontsize=9)
    plt.text(detail_x + detail_width + 0.1, detail_y + backsheet_thickness + eva_bottom_thickness + cell_thickness + eva_top_thickness + glass_thickness/2, 
             'Front Glass', va='center', ha='left', fontsize=9)

    plt.text(detail_x + detail_width/2, detail_y + detail_height + 0.2, 
             'Solar Panel Cross-Section', ha='center', fontsize=11, weight='bold')

    junction_box_x = rotated_corners[1][0] - 0.1
    junction_box_y = rotated_corners[1][1] - 0.2
    junction_box = Rectangle((junction_box_x, junction_box_y), 0.4, 0.3, 
                           angle=np.degrees(tilt_angle), 
                           edgecolor='black', facecolor='#8B4513')
    ax.add_patch(junction_box)
    plt.text(junction_box_x + 0.6, junction_box_y, 'Junction Box', fontsize=10)

    plt.annotate('Solar\nRadiation', xy=(rotated_corners[2][0] - 0.5, rotated_corners[2][1] + 0.3), 
                 xytext=(sun_x - 1, sun_y - 1),
                 arrowprops=dict(facecolor='orange', shrink=0.05, width=2, headwidth=8),
                 fontsize=10)

    plt.annotate('Electrical\nOutput', xy=(junction_box_x + 0.2, junction_box_y - 0.5), 
                 xytext=(junction_box_x + 1, junction_box_y - 1),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                 fontsize=10)

    plt.annotate('Heat\nLoss', xy=(rotated_corners[0][0] + 0.5, rotated_corners[0][1] + 0.5), 
                 xytext=(rotated_corners[0][0] - 1, rotated_corners[0][1] - 0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=10)

    temp_x, temp_y = rotated_corners[3][0] - 0.3, rotated_corners[3][1] - 0.3
    thermo = Rectangle((temp_x - 0.05, temp_y), 0.1, 0.5, 
                     edgecolor='black', facecolor='white')
    ax.add_patch(thermo)
    thermo_ball = Circle((temp_x, temp_y - 0.1), 0.15, 
                       edgecolor='black', facecolor='red')
    ax.add_patch(thermo_ball)
    plt.text(temp_x - 0.5, temp_y + 0.2, 'Cell\nTemperature', fontsize=10)

    wind_x, wind_y = -3, 3
    plt.text(wind_x, wind_y, "Wind", fontsize=12, weight='bold')
    for i in range(3):
        plt.arrow(wind_x + 0.8 + i*0.4, wind_y, 0.3, -0.15,
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        plt.arrow(wind_x + 0.8 + i*0.4, wind_y - 0.2, 0.3, -0.1,
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        plt.arrow(wind_x + 0.8 + i*0.4, wind_y + 0.2, 0.3, -0.2,
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    plt.title("Solar Panel Schematic with Energy Balance", fontsize=16)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig("solar_panel_schematic.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_IV_curve_plot():
    """
    Create and display I-V and P-V curves for different conditions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    T_amb_values = [298.15, 308.15, 318.15]  
    G_values = [600, 800, 1000]  

    colors = ['blue', 'green', 'red']

    for i, (T_amb, G) in enumerate(zip(T_amb_values, G_values)):
        T_cell = model.calculate_cell_temperature(T_amb, G)
        G_effective = model.calculate_effective_irradiance(G)

        V, I, P = model.calculate_IV_curve(T_cell, G_effective)

        label = f"T={T_amb-273.15:.1f}°C, G={G} W/m²"

        ax1.plot(V, I, color=colors[i], linewidth=2, label=label)
        ax2.plot(V, P, color=colors[i], linewidth=2, label=label)

        max_power_idx = np.argmax(P)
        ax1.plot(V[max_power_idx], I[max_power_idx], 'o', color=colors[i], markersize=8)
        ax2.plot(V[max_power_idx], P[max_power_idx], 'o', color=colors[i], markersize=8)

    ax1.set_title("I-V Curve under Different Conditions", fontsize=14)
    ax1.set_xlabel("Voltage (V)", fontsize=12)
    ax1.set_ylabel("Current (A)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.set_title("P-V Curve under Different Conditions", fontsize=14)
    ax2.set_xlabel("Voltage (V)", fontsize=12)
    ax2.set_ylabel("Power (W)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("IV_curve_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def simulate_daily_production(latitude=40.0, day_of_year=180, albedo=0.2, tilt=None):
    """
    Simulate solar panel production throughout a day
    """
    if tilt is None:
        tilt = params.tilt

    hours = np.linspace(5, 19, 29)  

    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    hour_angles = 15 * (hours - 12)  

    altitude_angles = np.arcsin(
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angles))
    )
    altitude_angles = np.degrees(altitude_angles)

    azimuth_angles = np.degrees(np.arccos(
        (np.sin(np.radians(altitude_angles)) * np.sin(np.radians(latitude)) - 
         np.sin(np.radians(declination))) /
        (np.cos(np.radians(altitude_angles)) * np.cos(np.radians(latitude)))
    ))

    azimuth_angles = np.where(hour_angles > 0, 360 - azimuth_angles, azimuth_angles)

    AOI = np.degrees(np.arccos(
        np.cos(np.radians(altitude_angles)) * np.cos(np.radians(azimuth_angles - params.azimuth)) * np.sin(np.radians(tilt)) +
        np.sin(np.radians(altitude_angles)) * np.cos(np.radians(tilt))
    ))

    AM = 1 / np.sin(np.radians(altitude_angles))
    AM = np.where(AM < 0, 999, AM)  

    G_dir = 1000 * np.exp(-0.13 * AM) * (altitude_angles > 0)
    G_diff = 150 * np.exp(-0.13 * AM) * (altitude_angles > 0)

    G_tilt = G_dir * np.cos(np.radians(AOI)) * (AOI < 90) + \
             G_diff * (1 + np.cos(np.radians(tilt))) / 2 + \
             (G_dir + G_diff) * albedo * (1 - np.cos(np.radians(tilt))) / 2

    T_amb = 293.15 + 8 * np.sin(np.pi * (hours - 5) / 14)  

    power_output = []
    cell_temps = []
    efficiencies = []

    for i, hour in enumerate(hours):
        if G_tilt[i] > 0:
            performance = model.solar_panel_performance(
                T_amb[i], G_tilt[i], 
                angle_of_incidence=AOI[i] if AOI[i] < 90 else 90,
                AM=AM[i] if AM[i] < 10 else 10
            )
            power_output.append(performance['P_electrical'])
            cell_temps.append(performance['T_cell'])
            efficiencies.append(performance['efficiency'])
        else:
            power_output.append(0)
            cell_temps.append(T_amb[i])
            efficiencies.append(0)

    return hours, power_output, cell_temps, efficiencies, G_tilt

def plot_daily_simulation():
    """
    Plot the results of daily solar panel simulation
    """
    seasons = [(80, 'Winter'), (170, 'Spring'), (260, 'Summer'), (350, 'Fall')]

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.3)

    for day, season in seasons:
        hours, power, temps, effs, irradiance = simulate_daily_production(day_of_year=day)

        axes[0].plot(hours, irradiance, label=season, linewidth=2)
        axes[1].plot(hours, np.array(temps) - 273.15, label=season, linewidth=2)
        axes[2].plot(hours, power, label=season, linewidth=2)

    axes[0].set_title("Solar Irradiance Throughout the Day", fontsize=14)
    axes[0].set_ylabel("Irradiance (W/m²)", fontsize=12)
    axes[0].set_xticks(np.arange(6, 20, 2))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].set_title("Cell Temperature Throughout the Day", fontsize=14)
    axes[1].set_ylabel("Temperature (°C)", fontsize=12)
    axes[1].set_xticks(np.arange(6, 20, 2))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    axes[2].set_title("Power Output Throughout the Day", fontsize=14)
    axes[2].set_xlabel("Hour of Day", fontsize=12)
    axes[2].set_ylabel("Power Output (W)", fontsize=12)
    axes[2].set_xticks(np.arange(6, 20, 2))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.savefig("daily_production_simulation.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_annual_production(latitude=40.0, tilt=None, azimuth=None):
    """
    Analyze annual solar panel production and create a heatmap
    """
    if tilt is None:
        tilt = params.tilt
    if azimuth is None:
        azimuth = params.azimuth

    days = np.arange(0, 365, 15)  
    hours = np.arange(5, 20, 0.5)  

    production_matrix = np.zeros((len(days), len(hours)))

    for i, day in enumerate(days):
        for j, hour in enumerate(hours):

            declination = 23.45 * np.sin(np.radians(360 * (284 + day) / 365))
            hour_angle = 15 * (hour - 12)

            altitude_angle = np.arcsin(
                np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
                np.cos(np.radians(hour_angle))
            )
            altitude_angle = np.degrees(altitude_angle)

            if altitude_angle > 0:

                AM = 1 / np.sin(np.radians(altitude_angle))
                G_dir = 1000 * np.exp(-0.13 * AM)
                G_diff = 150 * np.exp(-0.13 * AM)

                base_temp = 5 + 20 * np.sin(np.pi * day / 365)  
                hour_temp = base_temp + 8 * np.sin(np.pi * (hour - 5) / 14)  
                T_amb = 273.15 + hour_temp

                G_eff = 0.85 * (G_dir + G_diff)  

                if G_eff > 0:
                    performance = model.solar_panel_performance(T_amb, G_eff)
                    production_matrix[i, j] = performance['P_electrical']

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(production_matrix, cmap='viridis', aspect='auto', 
                   extent=[min(hours), max(hours), max(days), min(days)])

    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticks(month_days)
    ax.set_yticklabels(month_names)

    ax.set_xticks(np.arange(5, 20, 1))

    ax.set_title(f"Annual Solar Panel Power Output (W) - Latitude: {latitude}°, Tilt: {tilt}°", fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Month", fontsize=12)

    cbar = plt.colorbar(im)
    cbar.set_label('Power Output (W)', fontsize=12)

    plt.tight_layout()
    plt.savefig("annual_production_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    daily_energy = np.trapz(production_matrix, x=hours, axis=1) / 1000  
    annual_energy = np.sum(daily_energy) * 365 / len(days)  

    return annual_energy, daily_energy, days

def optimize_tilt():
    """
    Find optimal tilt angle for maximum annual production
    """
    latitude = 40.0  
    tilt_angles = np.arange(0, 61, 5)
    annual_energy = []

    for tilt in tilt_angles:
        energy, _, _ = analyze_annual_production(latitude=latitude, tilt=tilt)
        annual_energy.append(energy)

    plt.figure(figsize=(10, 6))
    plt.plot(tilt_angles, annual_energy, 'o-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title(f"Annual Energy Production vs. Tilt Angle at Latitude {latitude}°", fontsize=14)
    plt.xlabel("Tilt Angle (degrees)", fontsize=12)
    plt.ylabel("Annual Energy Production (kWh)", fontsize=12)

    optimal_tilt = tilt_angles[np.argmax(annual_energy)]
    max_energy = max(annual_energy)

    plt.annotate(f"Optimal Tilt: {optimal_tilt}°\nMax Energy: {max_energy:.1f} kWh", 
                xy=(optimal_tilt, max_energy), xytext=(optimal_tilt+5, max_energy-10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.savefig("tilt_optimization.png", dpi=300, bbox_inches='tight')
    plt.show()

    return optimal_tilt, max_energy
def save_output():

    plt.figure(figsize=(15, 12))
    create_plots()
    plt.savefig("solar_panel_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    create_schematic()
    plt.savefig("solar_panel_schematic.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))
    create_IV_curve_plot()
    plt.savefig("IV_curve_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 15))
    plot_daily_simulation()
    plt.savefig("daily_production_simulation.png", dpi=300, bbox_inches='tight')
    plt.close()

    annual_energy, daily_energy, days = analyze_annual_production()
    plt.savefig("annual_production_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    optimal_tilt, max_energy = optimize_tilt()
    plt.savefig("tilt_optimization.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_all_outputs(output_dir="./output_images/"):
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(15, 12))
    create_plots()
    plt.savefig(os.path.join(output_dir, "solar_panel_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    create_schematic()
    plt.savefig(os.path.join(output_dir, "solar_panel_schematic.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))
    create_IV_curve_plot()
    plt.savefig(os.path.join(output_dir, "IV_curve_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 15))
    plot_daily_simulation()
    plt.savefig(os.path.join(output_dir, "daily_production_simulation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    annual_energy, daily_energy, days = analyze_annual_production()
    plt.savefig(os.path.join(output_dir, "annual_production_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    optimal_tilt, max_energy = optimize_tilt()
    plt.savefig(os.path.join(output_dir, "tilt_optimization.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All images saved to {output_dir}")
    return annual_energy, optimal_tilt, max_energy

if __name__ == "__main__":
    create_plots()
    create_schematic()
    create_IV_curve_plot()
    plot_daily_simulation()
    save_output()
    save_all_outputs()
    annual_energy, _, _ = analyze_annual_production()
    print(f"Estimated Annual Energy Production: {annual_energy:.1f} kWh")
    optimal_tilt, max_energy = optimize_tilt()
    print(f"Optimal tilt angle: {optimal_tilt}° with annual production: {max_energy:.1f} kWh")