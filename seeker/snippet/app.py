#date: 2025-05-19T16:38:27Z
#url: https://api.github.com/gists/bdfdde3a061c3a62ff6b8e0a18c97929
#owner: https://api.github.com/users/mariomartgarcia

from shiny import App, render, ui
import numpy as np
from shiny.types import ImgData
from pathlib import Path




output_y = {
    "1": "High Sensitivity (0.989 &plusmn; 0.019, 0.550 &plusmn; 0.045)",
    "2": "Moderate Sensitivity (0.901 &plusmn; 0.018, 0.741 &plusmn; 0.024)",
    "3": "Balanced (0.825 &plusmn; 0.052, 0.816 &plusmn; 0.023)",
    "4": "Moderate Specificity (0.717 &plusmn; 0.022, 0.906 &plusmn; 0.010)",
    "5": "High Specificity (0.244 &plusmn; 0.017, 0.990 &plusmn; 0.003)"
}




#Weights
m_weight_alta_sensibilidad = np.array([-4.678601, -3.363664,  1.904342,  3.522168,  1.945306,
       -0.207524,  2.120496,  5.575156,  4.053366,  4.576220,
       -1.088831,  3.275743,  2.457869,  0.077084])

m_weight_media_sensibilidad = np.array([-3.589910, -1.923752,  1.527508,  2.801661,  1.946064,
        0.179350,  1.745069,  3.842988,  3.347437,  2.924218,
       -0.598635,  2.994618,  2.387489, -2.871972])

m_weight_equi = np.array([-3.964630, -1.178555,  1.724918,  1.858161,  1.771910,
        0.690755,  1.531704,  2.557498,  3.018703,  3.020277,
       -0.554450,  2.701213,  1.866281, -3.448869])

m_weight_media_especificidad = np.array([-3.094559, -1.150939,  1.291020,  1.929299,  1.681143,
        0.674711,  1.794956,  1.013513,  2.775508,  2.177254,
       -0.785831,  2.746239,  1.637317, -3.632753])

m_weight_alta_especificidad = np.array([-2.226893, -0.372100,  1.099336,  1.713687,  1.668148,
        0.727471,  1.647533,  0.901644,  2.786865,  1.900339,
       -0.872824,  2.723109,  1.308474, -5.737858])


app_ui = ui.page_fluid(ui.output_image( "image", inline = True),
    ui.tags.style("""
        /* Forzar que las etiquetas de los radio buttons no hagan wrap */
        .shiny-options-group label {
            white-space: nowrap !important;
        }
    
        /* Estilos para etiquetas de inputs */
        @media (min-width: 768px) {
            .input-label {
                font-weight: 600;
            }
        }
    
        .input-label {
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
            display: block;
        }
    
        /* Contenedor estético para cada sección */
        .form-section {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 1rem;
            background-color: #f8f9fa;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
    """),
    ui.column(10, 
        {"class": "col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"},
        # Title
        ui.h1(ui.HTML("<b>Stroke Mortality | Dynamic Models</b>"))),
        #ui.h6(ui.HTML("Calculadora pron&oacute;stica asociada a la publicaci&oacute;n <i>&quotComputaci&oacute;n hacia una medicina personalizada: optimizaci&oacute;n de modelos predictivos din&aacute;micos  en la toma de decisiones&quot</i> presentado en la LXXV Reuni&oacute;n Anual de la Sociedad Espa&ntilde;ola de Neurolog&iacute;a (Valencia, Noviembre 2023)." )),
        #ui.h6(ui.HTML("Juan Marta Enguita<sup>1,2</sup>, Mario Mart&iacute;nez-Garc&iacute;a<sup>3</sup>, Idoia Rubio Baines<sup>2,4</sup>, Mar&iacute;a Herrera Isasi<sup>2,4</sup>, Maite Mendioroz Iriarte<sup>2,4</sup>, Roberto Mu&ntilde;oz Arrondo<sup>2,4</sup>, Patricia de la Riva<sup>1</sup>, Maite Mart&iacute;nez Zabaleta<sup>1</sup>, I&ntilde;aki Inza<sup>5</sup>, Jose A. Lozano<sup>3,5</sup>")),
        #ui.h6(ui.HTML("<sup>1</sup>Servicio Neurolog&iacute;a. Hospital Universitario Donostia (HUD). <sup>2</sup>Grupo de investigaci&oacute;n en enfermedades cerebrovasculares. IdisNa, Pamplona. <sup>3</sup>Basque Center for Applied Mathemathics (BCAM), Bilbao, Espa&ntilde;a. <sup>4</sup>Servicio Neurolog&iacute;a. Hospital Universitario Navarra (HUN). <sup>5</sup>Universidad del Pa&iacute;s Vasco UPV/EHU, Facultad de Inform&aacute;tica, San Sebasti&aacute;n, Espa&ntilde;a.")))
        # input slider
                       


    ui.column(
                12,
                {"class": "col-lg-6 py-5 mx-auto"},
                ui.card(
                    ui.card_header(ui.HTML("<h5><b>Select Prediction Model (Sensitivity, Specificity)</b></h5>")),
                    ui.input_radio_buttons("y", " ", output_y)
                )
            ),
    ui.column(12, 
        {"class": "col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"},
        # Title
    ui.h5(ui.HTML("<b>Clinical features [Range] (Units*).</b>")),
    ui.h6(ui.HTML("*Features without units are represented by integers within the described range."))
        # input slider
    ),
        
    ui.div(
        {"style": "background-color: #f5f5f5; padding: 20px; border-radius: 8px;"},
        ui.row({"style": "margin-left:150px; margin-right:100px"},
            ui.column(4,
                ui.div(
                    ui.HTML('<label class="input-label">Platelets <small>[0–500] (10<sup>3</sup>/&#181;L)</small></label>'),
                    ui.input_numeric("x1", None, value=0, min=0, max=500)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Glomerular Filtration Rate <small>[0–120] (mL/min)</small></label>'),
                    ui.input_numeric("x2", None, value=0, min=0, max=120)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Infection at Admission <small>[0–1]</small></label>'),
                    ui.input_numeric("x3", None, value=0, min=0, max=1)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">C-Reactive Protein <small>[0–200] (mg/L)</small></label>'),
                    ui.input_numeric("x4", None, value=0, min=0, max=200)
                )
            ),
            ui.column(4,
                ui.div(
                    ui.HTML('<label class="input-label">Neutrophils <small>(10<sup>3</sup>/&#181;L)</small></label>'),
                    ui.input_numeric("x5", None, value=0, min=0, max=20)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Systolic Blood Pressure <small>[60–250] (mmHg)</small></label>'),
                    ui.input_numeric("x6", None, value=0, min=60, max=250)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Diastolic Blood Pressure <small>[40–150] (mmHg)</small></label>'),
                    ui.input_numeric("x7", None, value=0, min=40, max=150)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Blood Glucose <small>[40–600] (mg/dL)</small></label>'),
                    ui.input_numeric("x8", None, value=0, min=40, max=600)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Baseline Rankin Score <small>[0–5]</small></label>'),
                    ui.input_numeric("x9", None, value=0, min=0, max=5)
                )
            ),
            ui.column(4,
                ui.div(
                    ui.HTML('<label class="input-label">Hemorrhagic Transformation <small>[0–5]</small></label>'),
                    ui.input_numeric("x10", None, value=0, min=0, max=5)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Brain Ischemia Scale (ASPECTS) <small>[0–10]</small></label>'),
                    ui.input_numeric("x11", None, value=0, min=0, max=10)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">NIHSSb <small>[0–25]</small></label>'),
                    ui.input_numeric("x12", None, value=0, min=0, max=25)
                ),
                ui.div(
                    ui.HTML('<label class="input-label">Age <small>[18–100] (years)</small></label>'),
                    ui.input_numeric("x13", None, value=0, min=18, max=100)
                )
            )
        )
    ),
    ui.column(12, 
        {"class": "col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"},
        ui.h5(ui.HTML("<b>Patient Prognosis</b>"))),    
    ui.column(12,
        {"class": "col-lg-8 py-5 mx-auto"},
        ui.output_text_verbatim("txt")),
    ui.column(12, 
        {"class": "col-lg-8 py-5 mx-auto"},
        ui.p(
            ui.h5(ui.HTML("<b>References</b>")),
            ui.HTML("""
                <p>This calculator is the result of the work presented in <i>Dynamic Predictive Models in Personalized Medicine: A Multi-Objective Optimization Approach for Stroke Management</i>, submitted to the <i>International Journal of Stroke</i>.</p>
            """))))

def MinMaxScalerMort(X):
    data_min = np.array([55. ,  7. ,  0. ,  0. ,  0.5, 88. ,  7. , 50. ,  0. ,  0. ,  0. ,
        0. ,  0. ])
    data_max = np.array([770. ,  60. ,   1. , 304.1,  18.3, 257. , 144. , 575. ,   4. ,
         5. ,   1. ,   3. ,   3. ])
    x_norm = (X - data_min)/(data_max - data_min)
    return x_norm

def sigmoid(x, w):
    return 1 / (1 + np.exp(-np.matmul(w, x.T)))

def cat_ASPECT(x):
    if x <=7:
        return 0
    else:
        return 1
def cat_NIHSSb(x):
    if x <=5:
        return 0
    elif x<=14:
        return 1
    elif x<21:
        return 2
    else:
        return 3

def cat_edad(x):
    if x <=60:
        return 0
    elif x<=75:
        return 1
    elif x<=85:
        return 2
    else:
        return 3

        
def server(input, output, session):
    
    @output
    @render.image
    
    def image():

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logoall.svg"), "width": "800px", "style":"display: block; margin-left: auto; margin-right: auto;"}
        return img

        
    @output
    @render.text
    def txt():
        values_check = True
        cat_as = cat_ASPECT(input.x11())
        cat_NI = cat_NIHSSb(input.x12())
        cat_ed = cat_edad(input.x13())
     
        
        inputs = np.array([input.x1(), input.x2(), input.x3(), input.x4(), input.x5(), 
                           input.x6(), input.x7(), input.x8(), input.x9(), input.x10(),
                           cat_as, cat_NI, cat_ed])

        input_norm = np.append(MinMaxScalerMort(inputs), 1)

        error_test = "ERROR:"
        if input.x1() > 500 or input.x1() < 0:
            values_check = False
            error_test = error_test + "The valid range for Platelets is [0,500]. "
        
        if input.x2() > 120 or input.x2() < 0:
            values_check = False
            error_test = error_test + "The valid range for Glomerular Filtration is [0,120]. "
        
        if input.x3() > 1 or input.x3() < 0:
            values_check = False
            error_test = error_test + "Valid values for Infection at Admission are 0 or 1. "
        
        if input.x4() > 200 or input.x4() < 0:
            values_check = False
            error_test = error_test + "The valid range for CRP is [0,200]. "
        
        if input.x6() > 250 or input.x6() < 60:
            values_check = False
            error_test = error_test + "The valid range for Systolic Blood Pressure is [60, 250]. "
        
        if input.x7() > 150 or input.x7() < 40:
            values_check = False
            error_test = error_test + "The valid range for Diastolic Blood Pressure is [40, 150]. "
        
        if input.x8() > 600 or input.x8() < 40:
            values_check = False
            error_test = error_test + "The valid range for Blood Glucose is [40, 600]. "
        
        if input.x9() > 5 or input.x9() < 0:
            values_check = False
            error_test = error_test + "Valid values for Baseline Rankin are integers in [0,5]. "
        
        if input.x10() > 5 or input.x10() < 0:
            values_check = False
            error_test = error_test + "Valid values for Hemorrhagic Transformation are integers in [0,5]. "
        
        if input.x11() > 10 or input.x11() < 0:
            values_check = False
            error_test = error_test + "Valid values for ASPECT score are integers in [0,10]. "
        
        if input.x12() > 25 or input.x12() < 0:
            values_check = False
            error_test = error_test + "Valid values for NIHSSb are integers in [0,25]. "
        
        if input.x13() > 100 or input.x13() < 18:
            values_check = False
            error_test = error_test + "Valid values for Age are integers in [18,100]. "
        
        if values_check:
            if input.y() == '1':
                p = sigmoid(input_norm, m_weight_alta_sensibilidad)
                if p > 0.8:
                    return f"The probability of death is {np.round(p, 3)}. Extreme risk."
                if p > 0.5:
                    return f"The probability of death is {np.round(p, 3)}. High risk."
                else:
                    return f"The probability of death is {np.round(p, 3)}. Survival cannot be guaranteed. Try a more specific model."
        
            if input.y() == '2':
                p = sigmoid(input_norm, m_weight_media_sensibilidad)
                if p > 0.5:
                    return f"The probability of death is {np.round(p, 3)}. High risk."
                else:
                    return f"The probability of death is {np.round(p, 3)}. Survival cannot be guaranteed. Try a more specific model."
        
            if input.y() == '3':
                p = sigmoid(input_norm, m_weight_equi)
                if p > 0.5:
                    return f"The probability of death is {np.round(p, 3)}. Moderate risk of death. Consider a more sensitive model."
                else:
                    return f"The probability of death is {np.round(p, 3)}. Low risk of death. Consider a more specific model."
        
            if input.y() == '4':
                p = sigmoid(input_norm, m_weight_media_especificidad)
                if p > 0.5:
                    return f"The probability of death is {np.round(p, 3)}. Death cannot be confirmed. Try a more sensitive model."
                else:
                    return f"The probability of death is {np.round(p, 3)}. Low risk of death."
        
            if input.y() == '5':
                p = sigmoid(input_norm, m_weight_alta_especificidad)
                if p < 0.2:
                    return f"The probability of death is {np.round(p, 3)}. Minimal risk of death."
                if p < 0.5:
                    return f"The probability of death is {np.round(p, 3)}. Low risk of death."
                else:
                    return f"The probability of death is {np.round(p, 3)}. Death cannot be confirmed. Try a more sensitive model."
        
        elif sum(inputs) == 0:
            return f"Please enter the data in the calculator."
        
        else:
            return f"{error_test}"

            





app = App(app_ui, server, debug=True)
