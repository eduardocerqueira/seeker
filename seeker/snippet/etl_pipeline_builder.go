//date: 2024-05-01T16:50:36Z
//url: https://api.github.com/gists/246dd5ca81b1f8e5a466f9942aee2e97
//owner: https://api.github.com/users/Harryalways317

//BASE ETL CLASS AND HELPER TO STORE THE STATE TO DB
type ETL struct {
	Name string
}

func (e *ETL) StoreStateToDB(state map[string]interface{}) {
	// logic to store the state to mongo or smth
	log.Printf("Storing state to database: %v", state)
}

// EACH STEP ONE BY ONE CAN BE CHAINED all implements the Step base class
type UploadStep struct {
	ETL
	DocStore DocumentStore
}

func (s *UploadStep) Execute(input map[string]interface{}) map[string]interface{} {
	bytes := input["bytes"].([]byte)
	docURL := s.DocStore.Upload(bytes)
	input["docURL"] = docURL
	s.StoreStateToDB(input)
	return input
}

func (s *UploadStep) Name() string {
	return "Upload"
}

type ProcessStep struct {
	ETL
	DocProcessor DocumentProcessor
}

func (s *ProcessStep) Execute(input map[string]interface{}) map[string]interface{} {
	docURL := input["docURL"].(string)
	content := s.DocProcessor.Process(docURL)
	input["content"] = content
	s.StoreStateToDB(input)
	return input
}

func (s *ProcessStep) Name() string {
	return "Process"
}


type ContentFillStep struct {
	ETL
	ContentFiller ContentFiller
}

func (s *ContentFillStep) Execute(input map[string]interface{}) map[string]interface{} {
	content := input["content"].(string)
	filledContent := s.ContentFiller.Fill(content)
	input["filledContent"] = filledContent
	s.StoreStateToDB(input)
	return input
}

func (s *ContentFillStep) Name() string {
	return "ContentFill"
}


type SummarizeStep struct {
	ETL
	Summarizer Summarizer
}

func (s *SummarizeStep) Execute(input map[string]interface{}) map[string]interface{} {
	filledContent := input["filledContent"].(string)
	summary := s.Summarizer.Summarize(filledContent)
	input["summary"] = summary
	return input
}

func (s *SummarizeStep) Name() string {
	return "Summarize"
}


type CreateEmbeddingsStep struct {
	ETL
	Embeddings Embeddings
}

func (s *CreateEmbeddingsStep) Execute(input map[string]interface{}) map[string]interface{} {
	filledContent := input["filledContent"].(string)
	embedding := s.Embeddings.Create(filledContent)
	input["embedding"] = embedding
	return input
}

func (s *CreateEmbeddingsStep) Name() string {
	return "CreateEmbeddings"
}




// PIPELINE and PIPELINE BUILDER CLASS will be going to client.go(in weavite terms) or even pipeline.go
type Pipeline struct {
	ETL
	Steps []Step
	Name  string
}

func (p *Pipeline) Execute(input map[string]interface{}) map[string]interface{} {
	for _, step := range p.Steps {
		log.Printf("Executing step: %s", step.Name())
		input = step.Execute(input)
		log.Printf("Step %s completed. Output: %v", step.Name(), input)
	}
	return input
}

func (p *Pipeline) Restore(input map[string]interface{}, restoreStep string) map[string]interface{} {
	for _, step := range p.Steps {
		if step.Name() == restoreStep {
			log.Printf("Restoring from step: %s", step.Name())
			input = step.Execute(input)
			log.Printf("Step %s completed. Output: %v", step.Name(), input)
		}
	}
	return input
}

type PipelineBuilder struct {
	pipeline *Pipeline
}

func NewPipelineBuilder(name string) *PipelineBuilder {
	return &PipelineBuilder{
		pipeline: &Pipeline{ETL: ETL{Name: name}},
	}
}

func (b *PipelineBuilder) AddStep(step Step) *PipelineBuilder {
	b.pipeline.Steps = append(b.pipeline.Steps, step)
	return b
}

func (b *PipelineBuilder) Build() *Pipeline {
	return b.pipeline
}


func main() {
	//  implementations of indivudial ones
	docStore := OisterDocumentStore() 
	docProcessor := OisterDocumentProcessor()
	contentFiller := OisterContentFiller() 
	summarizer := OisterSummarizer() 
	embeddings := OisterEmbeddings() 

	// steps
	uploadStep := &UploadStep{ETL: ETL{Name: "Upload"}, DocStore: docStore}
	processStep := &ProcessStep{ETL: ETL{Name: "Process"}, DocProcessor: docProcessor}
	contentFillStep := &ContentFillStep{ETL: ETL{Name: "ContentFill"}, ContentFiller: contentFiller}
	summarizeStep := &SummarizeStep{ETL: ETL{Name: "Summarize"}, Summarizer: summarizer}
	createEmbeddingsStep := &CreateEmbeddingsStep{ETL: ETL{Name: "CreateEmbeddings"}, Embeddings: embeddings}

	// pipeline creation
	pipeline := NewPipelineBuilder("OisterPipeline").
		AddStep(uploadStep).
		AddStep(processStep).
		AddStep(contentFillStep).
		AddStep(summarizeStep).
		AddStep(createEmbeddingsStep).
		Build()

	// Execution of pipeline
	input := map[string]interface{}{
		"bytes": []byte("Document content"),
	}
	output := pipeline.Execute(input)


	//Restore thing, can be done from step, as we initially stored the steps we can do it from that to end
	restoreOutput := pipeline.Restore(input, "Upload")
	restoreOutputBytes, _ := json.Marshal(restoreOutput)
	fmt.Println(string(restoreOutputBytes))

}