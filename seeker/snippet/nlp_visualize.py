#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

﻿import spacy
from visualize.visualize import visualize_user_activity_by_date, visualize_frequency, visualize_heatmap

nlp = spacy.load("en_core_web_sm")

# Dictionary mapping visualization types to functions
VISUALIZATION_FUNCTIONS = {
    "activity": visualize_user_activity_by_date,
    "heatmap": visualize_heatmap,
    "frequency": visualize_frequency
}

def parse_nlp_visualize(query):
    doc = nlp(query)
    
    visualization_type = None
    fields = []
    group_by = None
    
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"d "**********"o "**********"c "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"t "**********"e "**********"x "**********"t "**********". "**********"l "**********"o "**********"w "**********"e "**********"r "**********"( "**********") "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"" "**********"a "**********"c "**********"t "**********"i "**********"v "**********"i "**********"t "**********"y "**********"" "**********", "**********"  "**********"" "**********"u "**********"s "**********"a "**********"g "**********"e "**********"" "**********", "**********"  "**********"" "**********"e "**********"n "**********"g "**********"a "**********"g "**********"e "**********"m "**********"e "**********"n "**********"t "**********"" "**********", "**********"  "**********"" "**********"h "**********"e "**********"a "**********"t "**********"m "**********"a "**********"p "**********"" "**********", "**********"  "**********"" "**********"f "**********"r "**********"e "**********"q "**********"u "**********"e "**********"n "**********"c "**********"y "**********"" "**********", "**********"  "**********"" "**********"s "**********"t "**********"a "**********"t "**********"u "**********"s "**********"" "**********"] "**********": "**********"
            visualization_type = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"t "**********"e "**********"x "**********"t "**********". "**********"l "**********"o "**********"w "**********"e "**********"r "**********"( "**********") "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"" "**********"b "**********"y "**********"" "**********", "**********"  "**********"" "**********"p "**********"e "**********"r "**********"" "**********", "**********"  "**********"" "**********"g "**********"r "**********"o "**********"u "**********"p "**********"" "**********"] "**********": "**********"
            group_by = "**********"== "NOUN" and token.i > doc.index(token)), None)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"p "**********"o "**********"s "**********"_ "**********"  "**********"= "**********"= "**********"  "**********"" "**********"N "**********"O "**********"U "**********"N "**********"" "**********": "**********"
            fields.append(token.text)
    
    return {
        'type': visualization_type,
        'fields': fields,
        'group_by': group_by
    }

def handle_nlp_visualize(query, data):
    params = parse_nlp_visualize(query)
    if params and params['type'] in VISUALIZATION_FUNCTIONS:
        visualization_func = VISUALIZATION_FUNCTIONS[params['type']]
        try:
            visualization_func(data, grouping=params.get('group_by'))
            return f"Visualized {params['type']} by {params.get('group_by', 'default')}"
        except KeyError:
            return f"Required grouping not specified for {params['type']} visualization."
        except Exception as e:
            return f"An error occurred while visualizing: {str(e)}"
    else:
        return "Visualization not understood or not implemented yet."

def feedback_on_visualization(query):
    params = parse_nlp_visualize(query)
    if params:
        feedback = f"Interpreted request as: Visualize {params['type']} by {params.get('group_by', 'not specified')}"
    else:
        feedback = "Could not interpret the visualization request."
    print(feedback)

# Example usage, assuming you've loaded the data elsewhere
# feedback_on_visualization("show me user activity by month")
# handle_nlp_visualize("show me user activity by month", data)