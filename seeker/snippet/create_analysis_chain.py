#date: 2025-04-04T16:52:25Z
#url: https://api.github.com/gists/77c049de7189ddfd6361b6068ddfce26
#owner: https://api.github.com/users/h-swathi-shenoy

def create_invoice_analysis_chain():
    '''
    Chain includes following sequence
    1. extract_context : text from extractpdf function which further uses Mistral OCR Capablities
    2. analyze_document : analyze the invoice to extract fields as mentioned in prompt
    3. validate_invoice: validate if the key fields are present from analyze document state, which makes a it a valid invoice
    
    Orchestrate all the above steps in sequence via LangGraph Framework
    '''

    def extract_context(state: InvoiceAnalysisState):
        print("----------------------------------------------------")
        print("-----------Extracting context from PDF--------------")
        print("----------------------------------------------------")
        state['context'] = text
        return state
    
    def analyze_document(state: InvoiceAnalysisState):
        print("----------------------------------------------------")
        print("-----------Analyzing context from PDF--------------")
        print("----------------------------------------------------")
        messages = state['context']
        document_content = messages
        
    # Use Amazon NovaPro for invoice analysis
        messages = [
            SystemMessage(content="""You are a invoice document analyzer. Extract key information and format it in markdown with the following sections:

                ### Seller Details
                - Seller Name
                - Seller Address
        
                ### Buyer Details
                - Name of the buyer
                - Buyer Address
                - GST Number
                - PAN Number
                - State/UT Code

                ### Order Details
                  - Order Number
                  - Order Date

                ### Invoice Details
                - Invoice Number 
                - Invoice Details
                - Invoice Date
                
                ### Item Details
                -  List of Items with Description and Price without and with tax
                -  Total Amount
                
                ## Signature Details( Provide Yes if signed, else No)
                - Is it signed ? 

                ### Payment Details
                - List of Transaction Ids
                - List of Timestamp for transactio
                - Mode of Payment
                - Invoice Value
                Please ensure the response is well-formatted in markdown with appropriate headers and bullet points."""),
            HumanMessage(content=document_content)
        ]
        response = analyzer_llm.invoke(messages)
        
        state["analysis_result"] = response.content.split("</think>")[-1]
        return state


    def validate_invoice(state:InvoiceAnalysisState):
        print("----------------------------------------------------")
        print("------------Validating invoice from PDF-----------")
        print("----------------------------------------------------")
        analysis_result = state["analysis_result"]
        
        messages = [
            SystemMessage(content="""You are a invoice  validator. Provide your assessment in markdown format with these sections:

                ### Invoice Analysis
                - Check if there is invoice number in invoice
                - Check if there is Seller/ Billing Details
                - Evaluate if there GST Number/PAN Number in invoice
                - Table exists with item details and total amount
                - Check if payment details exist along with transaction id with invoice value

                Please format your response in clear markdown with appropriate headers and bullet points."""),
            HumanMessage(content=f"""Analysis: {analysis_result}
                         Based on the Analysis  provided please provide whether given invoice is a valid.
                         If not mention its not valid invoice, and also mention what details are missing from the invoice.
                         """)
        ]
        response = analyzer_llm.invoke(messages)
        
        state["validation_result"] = response.content.split("</think>")[-1]
        return state
    
    workflow = StateGraph(InvoiceAnalysisState)

    # Add nodes
    workflow.add_node("extractor", extract_context)
    workflow.add_node("analyzer", analyze_document)
    workflow.add_node("validator", validate_invoice)

    # Define edges
    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "analyzer")  
    workflow.add_edge("analyzer", "validator")
    workflow.add_edge("validator", END)


    # Compile the graph
    chain = workflow.compile()
    
    # Generate graph visualization
    graph_png = chain.get_graph().draw_mermaid_png()
    graph_base64 = base64.b64encode(graph_png).decode('utf-8')
    
    return chain, graph_base64