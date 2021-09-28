//date: 2021-09-28T17:06:11Z
//url: https://api.github.com/gists/fda3f49810368152dbadd40782884fbd
//owner: https://api.github.com/users/diogopeixoto

public class Compiler {
 
    private CodeGenerator codeGenerator;
    private Parser parser;
    private ProgramNodeBuilder programNodeBuilder;
    private Scanner scanner;
    
    public Compiler(CodeGenerator codeGenerator, Parser parser, ProgramNodeBuilder programNodeBuilder, Scanner scanner){
        this.codeGenerator = codeGenerator;
        this.parser = parser;
        this.programNodeBuilder = programNodeBuilder;
        this.scanner = scanner;
    }

    public void compile(Stream stream){
        // Lógica que vai chamar os métodos das classes CodeGenerator, Parser, ProgramNodeBuilder e Scanner
    }
}

public class CodeGenerator {
    // Lógica do subsistema responsável por gerar o código
}

public class Scanner {
    // Lógica do subsistema responsável por ler todo o Stream
}

public class Parser {
    // Lógica do subsistema responsável por transformar a entrada em tokens da linguagem de programação
}

public class ProgramNodeBuilder {
    // Lógica do subsistema responsável por construir a árvore de execução do programa
}