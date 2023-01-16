//date: 2023-01-16T16:43:32Z
//url: https://api.github.com/gists/d30c4304f91c365185f1e753814a3dae
//owner: https://api.github.com/users/sofeikov

    private static final int problemDim = 1;
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        SingleOutputStreamOperator<DenseVector> dataStream = env
                .socketTextStream("localhost", 9999)
                .map(new Splitter(problemDim))
                .countWindowAll(8)
                .process(new RegressionWindowFunction(problemDim));
        dataStream.print();
        env.execute("Training");
    }