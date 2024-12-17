//date: 2024-12-17T16:59:02Z
//url: https://api.github.com/gists/d460aed6ea6674122d95e012f704f628
//owner: https://api.github.com/users/antonino3g

package br.com.qg.gatewaybiro.executorconsulta;

import java.util.Arrays;

import br.com.qg.gatewaybiro.bilhete.TipoBilhete;
import br.com.qg.gatewaybiro.util.constant.RestServiceConstants;

public enum OpcaoConsulta {

    QUOD_PF("Quod PF", "QUODPF", "quod", null),
    QUOD_PJ("Quod PJ", "QUODPJ", "quod", null),
    QUOD_RENDA_PRESUMIDA("Quod Renda Presumida", "QUODRENDAPRESUMIDAPF", "presumedIncome", TipoBilhete.QUOD_RENDAPRESUMIDA),
    PROTESTO("Protesto", "PROTESTO", "protesto", TipoBilhete.PROTESTO),
    QUOD_CCF("Quod CCF", "QUODCCF", "quodCcf", TipoBilhete.QUOD_CCF),
    QUOD_MAIS_NEGOCIO_PF("Quod Mais Negócio PF", "QUODMAISNEGOCIOPF", "quodMaisNegocio", TipoBilhete.QUOD_MAISNEGOCIOPF),
    QUOD_FRAUDE_SCORE_PF("Quod Fraude Score PF", "QUODFRAUDESCOREPF", "quodFraudeScore", TipoBilhete.QUOD_FRAUDESCOREPF),
    QUOD_SCORE_PF("Quod Score PF", "QUODSCOREPF", "quodScore", TipoBilhete.QUOD_SCORE_PF),
    QUOD_SCORE_PJ("Quod Score PJ", "QUODSCOREPJ", "quodScore", TipoBilhete.QUOD_SCORE_PJ),
    QUOD_INADIMPLENCIA_PF("QUOD Inadimplência PF", "QUODINADIMPLENCIAPF", "quodInadimplencia", TipoBilhete.QUOD_NEGATIVACAO),
    QUOD_FATURAMENTO_PRESUMIDO("Quod Faturamento Presumido", "QUODPJFATURAMENTOPRESUMIDO", "presumido", TipoBilhete.QUOD_FATURAMENTOPRESUMIDO),
    QUOD_GRAU_ATIVIDADE_PJ("Quod Grau Atividade PJ", "QUODGRAUATIVIDADEPJ", "quodGrauAtividade", TipoBilhete.QUOD_GRAUATIVIDADEPJ),
    QUOD_SCORE_BOLETO_PJ("Quod Score Boleto PJ", "QUODPJSCOREBOLETO", "quodScoreBoleto", TipoBilhete.QUOD_SCOREBOLETO),
    QUOD_MAIS_POSITIVO_PJ("Quod Mais Positivo PJ", "QUODPJMAISPOSITIVO", "quodMaisPositivo", TipoBilhete.QUOD_MAISPOSITIVO),
    QUOD_MAIS_NEGOCIO_PJ("Quod Mais Negócio PJ", "QUODMAISNEGOCIOPJ", "quodMaisNegocio", TipoBilhete.QUOD_MAISNEGOCIO),
    QUOD_DETECTA_RISCO_PJ("Quod Detecta Risco PJ", "QUODPJDETECTARISCO", "quodDetectaRisco", TipoBilhete.QUOD_DETECTARISCO),
    QUOD_GASTO_ESTIMADO_PJ("Quod Gasto Estimado PJ", "QUODPJGASTOESTIMADO", "quodGastoEstimado", TipoBilhete.QUOD_GASTOESTIMADO),
    QUOD_INDICADORES_PF("Quod Indicadores PF", "QUODPFINDICADORESNEGOCIO", "quodIndicadores", TipoBilhete.QUOD_INDICADORES),
    QUOD_INDICADORES_PJ("Quod Indicadores PJ", "QUODPJINDICADORESNEGOCIO", "quodIndicadores", TipoBilhete.QUOD_INDICADORES),
    QUOD_SCORE_CUSTOM_PJ("Quod Score Custom", "QUODPJSCOREBRASILCUSTOM", "quodScoreCustom", TipoBilhete.QUOD_SCORECUSTOMPJ),
    QUOD_INDICADORES_400_PF("Quod Indicadores 400 PF", "QUODPFINDICADORESNEGOCIO400", "quodIndicadores400", TipoBilhete.QUOD_INDICADORES400),
    CADASTRAL_LOCAL("Cadastral", "CADASTRAL_LOCAL", "cadastral", TipoBilhete.CADASTRAL),
    CADASTRAL_LOCAL_PJ("Cadastral PJ", "CADASTRAL_LOCAL_PJ", "cadastral", TipoBilhete.CADASTRAL),
    QUOD_VERIFIQ_PJ("Quod Verifq PJ", "QUODVERIFIQPJ", "quodVerifiqPJ", TipoBilhete.QUOD_VERIFIQPJ),
    QUOD_LIMITE_CREDITO_PJ("Quod Limite Crédito PJ", "QUODPJLIMITECREDITO", "quodLimiteCredito", TipoBilhete.QUOD_LIMITECREDITO),
    QUOD_MAIS_CONTATO("Quod Mais Contato", "QUODMAISCONTATO", "quodMaisContato", TipoBilhete.QUOD_MAISCONTATO),
    PENDENCIA_TOOLSDATA("Pendência", "PENDENCIA_TOOLSDATA", "pendencia", TipoBilhete.PEDENCIA_TOOLSDATA),
    PROTESTO_TOOLSDATA("Protesto", "PROTESTO_TOOLSDATA", "protesto", TipoBilhete.PROTESTO_TOOLSDATA),
    CONSULTAAUTO_AGREGADO("Automotiva Agregado", "AUTOMOTIVA_AGREGADO", "automotivaAgregado", TipoBilhete.CONSULTAAUTO_AGREGADO),
    CONSULTAAUTO_PRECIFICADOR("Automotiva Precificador", "AUTOMOTIVA_PRECIFICADOR", "automotivaPrecificador", TipoBilhete.CONSULTAAUTO_PRECIFICADOR),
    CONSULTAAUTO_GRAVAMESIMPLES("Automotiva Gravame Simples", "AUTOMOTIVA_GRAVAMESIMPLES", "automotivaGravameSimples", TipoBilhete.CONSULTAAUTO_GRAVAMESIMPLES),
    //	CONSULTAAUTO_HISTORICOPROPRIETARIO("Automotiva Histórico Proprietários", "AUTOMOTIVA_HISTORICOPROPRIETARIO", "automotivaHistoricoProprietario", TipoBilhete.CONSULTAAUTO_HISTORICOPROPRIETARIO),
    CONSULTAAUTO_HISTORICOPROPRIETARIO("Automotiva Histórico Proprietários", "AUTOMOTIVA_HISTORICOPROPRIETARIO",
            RestServiceConstants.ConsultaAuto.AUTOMOTIVA_HISTORICOPROPRIETARIO, TipoBilhete.CONSULTAAUTO_HISTORICOPROPRIETARIO),
    CONSULTAAUTO_MOTOR("Automotiva Motor", "AUTOMOTIVA_MOTOR", "automotivaMotor", TipoBilhete.CONSULTAAUTO_MOTOR),
    CONSULTAAUTO_RENAINF("Automotiva Renainf", "AUTOMOTIVA_RENAINF", "automotivaRenainf", TipoBilhete.CONSULTAAUTO_RENAINF),
    CONSULTAAUTO_RENAJUD("Automotiva Renajud", "AUTOMOTIVA_RENAJUD", "automotivaRenajud", TipoBilhete.CONSULTAAUTO_RENAJUD),
    CONSULTAAUTO_BASEESTADUAL("Automotiva Base Estadual", "AUTOMOTIVA_BASEESTADUAL", "automotivaBaseEstadual", TipoBilhete.CONSULTAAUTO_BASEESTADUAL),
    CONSULTAAUTO_BASERENAVAM("Automotiva Base Renavam", "AUTOMOTIVA_BASERENAVAM", "automotivaBaseRenavam", TipoBilhete.CONSULTAAUTO_BASERENAVAM),
    CONSULTAAUTO_LEILAO("Automotiva Leilão", "AUTOMOTIVA_LEILAO", "automotivaLeilao", TipoBilhete.CONSULTAAUTO_LEILAO),
    CONSULTAAUTO_DECODIFICADOR("Automotiva Decodificador", "AUTOMOTIVA_DECODIFICADOR", "automotivaDecodificador", TipoBilhete.CONSULTAAUTO_DECODIFICADOR),
    CONSULTAAUTO_ONLINE("Automotiva Online", "AUTOMOTIVA_ONLINE", "automotivaOnline", TipoBilhete.CONSULTAAUTO_ONLINE),
    CONSULTAAUTO_PERDATOTAL("Automotiva Perda Total", "AUTOMOTIVA_PERDATOTAL", "automotivaPerdaTotal", TipoBilhete.CONSULTAAUTO_PERDATOTAL),
    CREDIFY_PF_NOME("Cadastral PF Nome", "CADASTRAL_PF_NOME",
            RestServiceConstants.Credify.CADASTRAL_NOME_PF, TipoBilhete.CREDIFY_PF_NOME),
    CREDIFY_PF_PESQUISA("Cadastral PF Pesquisa", "CADASTRAL_PF_PESQUISA",
            RestServiceConstants.Credify.CADASTRAL_PESQUISA_PF, TipoBilhete.CREDIFY_PF_PESQUISA),
    CREDIFY_PJ_PESQUISA("Cadastral PJ Pesquisa", "CADASTRAL_PJ_PESQUISA",
            RestServiceConstants.Credify.CADASTRAL_PESQUISA_PJ, TipoBilhete.CREDIFY_PJ_PESQUISA),
    CREDIFY_PJ_RAZAO_SOCIAL("Cadastral PJ Razão Social", "CADASTRAL_PJ_RAZAO_SOCIAL",
            RestServiceConstants.Credify.CADASTRAL_RAZAO_SOCIAL_PJ, TipoBilhete.CREDIFY_PJ_RAZAO_SOCIAL),
    CREDIFY_PF_TELEFONE("Cadastral PF Telefone", "CADASTRAL_PF_TELEFONE",
            RestServiceConstants.Credify.CADASTRAL_TELEFONE_PF, TipoBilhete.CREDIFY_PF_TELEFONE),
    CREDIFY_PJ_TELEFONE("Cadastral PJ Telefone", "CADASTRAL_PJ_TELEFONE",
            RestServiceConstants.Credify.CADASTRAL_TELEFONE_PJ, TipoBilhete.CREDIFY_PJ_TELEFONE),
    CREDIFY_PF_EMAIL("Cadastral PF Email", "CADASTRAL_PF_EMAIL",
            RestServiceConstants.Credify.CADASTRAL_EMAIL_PF, TipoBilhete.CREDIFY_PF_EMAIL),
    CREDIFY_PJ_EMAIL("Cadastral PJ Email", "CADASTRAL_PJ_EMAIL",
            RestServiceConstants.Credify.CADASTRAL_EMAIL_PJ, TipoBilhete.CREDIFY_PJ_EMAIL),
    CREDIFY_PROPR_TELEFONE_OPERADORA("Cadastral Proprietário Telefone Operadora", "CADASTRAL_PROPR_TELEFONE_OPERADORA",
            RestServiceConstants.Credify.CADASTRAL_PROPR_TELEFONE_OPERADORA, TipoBilhete.CREDIFY_PROPR_TELEFONE_OPERADORA),
    CREDIFY_CNH_DOCUMENTO("Cadastral CNH Documento", "CADASTRAL_CNH_DOCUMENTO",
            RestServiceConstants.Credify.CADASTRAL_DOCUMENTO_CNH, TipoBilhete.CREDIFY_CNH_DOCUMENTO),
    CREDIFY_PRECIFICADOR_FIPE("Automotiva Precificador Fipe", "AUTO_PRECIFICADOR_FIPE",
            RestServiceConstants.Credify.AUTO_PRECIFICADOR_FIPE, TipoBilhete.CREDIFY_PRECIFICADOR_FIPE),
    CREDIFY_VEICULO_DOCUMENTO("Automotiva Veículo por Documento", "AUTO_VEICULO_DOCUMENTO",
            RestServiceConstants.Credify.AUTO_VEICULO_DOCUMENTO, TipoBilhete.CREDIFY_VEICULO_DOCUMENTO),
    CREDIFY_VEICULO_DOCUMENTO_FROTA("Automotiva Veículo Documento Frota", "AUTO_VEICULO_DOCUMENTO_FROTA",
            RestServiceConstants.Credify.AUTO_VEICULO_DOCUMENTO_FROTA, TipoBilhete.CREDIFY_VEICULO_DOCUMENTO_FROTA),
    CREDIFY_CPF_CNPJ_TOTALIZADOR("Totalizador Processos CPF / CNPJ", "CPF_CNPJ_TOTALIZADOR_PROCESSOS",
            RestServiceConstants.Credify.CPF_CNPJ_TOTALIZADOR_PROCESSOS, TipoBilhete.CREDIFY_CPF_CNPJ_TOTALIZADOR),
    GPS_9000_391("GPS 9000 391", "GPS_9000_391", "gps391", TipoBilhete.GPS_9000_391),
    GPS_9000_353("GPS 9000 353", "GPS_9000_353", "gps353", TipoBilhete.GPS_9000_353),
    GPS_9000_627("GPS 9000 627", "GPS_9000_627", "gps627", TipoBilhete.GPS_9000_627),
    GPS_9999_894("GPS 9999 894", "GPS_9999_894", "gps894", TipoBilhete.GPS_9999_894),
    GPS_9999_895("GPS 9999 895", "GPS_9999_895", "gps895", TipoBilhete.GPS_9999_895),
    AUTOCORP_LEILAO("leilao_007", "AUTOCORP_LEILAO", "leilao_007", TipoBilhete.AUTOCORP_LEILAO),
    DATESOLUTIONS_DADOS_PF("Dados Pessoa Física", "DADOS_PESSOA_FISICA",
            RestServiceConstants.DateSolutions.DADOS_PF, TipoBilhete.DATESOLUTIONS_PF),
    DATESOLUTIONS_DADOS_PJ("Dados Pessoa Jurídica", "DADOS_PESSOA_JURIDICA",
            RestServiceConstants.DateSolutions.DADOS_PJ, TipoBilhete.DATESOLUTIONS_PJ),
    DATESOLUTIONS_CREDITO_SIMPLES("Crédito Simples", "CREDITO_SIMPLES",
            RestServiceConstants.DateSolutions.CREDITO_SIMPLES, TipoBilhete.DATESOLUTIONS_SIMPLES),
    DATESOLUTIONS_CREDITO_COMPLETA("Crédito Completa", "CREDITO_COMPLETA",
            RestServiceConstants.DateSolutions.CREDITO_COMPLETA, TipoBilhete.DATESOLUTIONS_COMPLETA),
    APIDECONSULTA_SCORE_TURBO("Score Turbo", "SCORE_TURBO",
            RestServiceConstants.ApiDeConsulta.SCORE_TURBO, TipoBilhete.APIDECONSULTA_SCORE_TURBO),
    APIDECONSULTA_PGFN("PGFN Documento", "PGFN_DOCUMENTO",
            RestServiceConstants.ApiDeConsulta.PGFN, TipoBilhete.APIDECONSULTA_PGFN),
    CREDAUTO_HISTORICO_ROUBO_FURTO("Histórico Roubo e Furto", "HISTORICO_ROUBO_FURTO",
            RestServiceConstants.Credauto.HISTORICO_ROUBO_FURTO, TipoBilhete.CREDAUTO_HISTORICO_ROUBO_FURTO),
    CREDAUTO_BASE_LEILAO("Base Leilão", "BASE_LEILAO",
            RestServiceConstants.Credauto.BASE_LEILAO, TipoBilhete.CREDAUTO_BASE_LEILAO),
    KARFEX("leilao_006", "KARFEX", "karfex", TipoBilhete.KARFEX),
    CHECKPRICE_RENAINF("Automotiva Renainf", "AUTO_RENAINF",
            RestServiceConstants.Checkprice.RENAINF, TipoBilhete.CHECKPRICE_RENAINF),
    CHECKPRICE_CRLV("Automotiva Crlv", "AUTO_CRLV",
            RestServiceConstants.Checkprice.CRLV, TipoBilhete.CHECKPRICE_CRLV),
    MTIX_VEICULO_PROPRIETARIO("Automotiva Veículo Proprietário", "AUTO_VEICULO_PROPRIETARIO",
            RestServiceConstants.Mtix.VEICULO_PROPRIETARIO, TipoBilhete.MTIX_VEICULO_PROPRIETARIO),
    MTIX_NACIONAL_CSV("Automotiva Csv", "AUTO_CSV",
            RestServiceConstants.Mtix.CSV, TipoBilhete.MTIX_NACIONAL_CSV),
    PLACA_FIPE("Placa Fipe", "PLACA_FIPE",
            RestServiceConstants.PlacaFipe.PLACA_FIPE, TipoBilhete.PLACA_FIPE),
    INFOSIMPLES_CNDT("CNDT", "CNDT", RestServiceConstants.InfoSimples.CNDT, TipoBilhete.CNDT),

    NEXO_SENATRAN_CONDUTOR("Automotiva Senatran Condutor", "AUTO_SENATRAN_CONDUTOR",
            RestServiceConstants.Nexo.SENATRAN_CONDUTOR, TipoBilhete.NEXO_SENATRAN_CONDUTOR),
    NEXO_CSV_COMPLETO("Automotiva CSV Completo", "AUTO_CSV_COMPLETO",
            RestServiceConstants.Nexo.CSV_COMPLETO, TipoBilhete.NEXO_CSV_COMPLETO),
    NEXO_PROPRIETARIO_VEICULAR("Automotiva Proprietário Veicular", "AUTO_PROPRIETARIO_VEICULAR",
            RestServiceConstants.Nexo.PROPRIETARIO_VEICULAR, TipoBilhete.NEXO_PROPRIETARIO_VEICULAR),
    NEXO_CNH_RETRATO("Automotiva Retrato CNH", "AUTO_CNH_RETRATO",
            RestServiceConstants.Nexo.CNH_RETRATO, TipoBilhete.NEXO_CNH_RETRATO),
    NEXO_PLACA_CRV_PDF("Automotiva Placa CRV", "AUTO_PLACA_CRV_PDF",
            RestServiceConstants.Nexo.PLACA_CRV_PDF, TipoBilhete.NEXO_PLACA_CRV_PDF),
    NEXO_CNH_COD("Automotiva Código Segurança CNH", "AUTO_CNH_COD",
            RestServiceConstants.Nexo.CNH_COD, TipoBilhete.NEXO_CNH_COD),
    NEXO_IMAGENS_LEILAO("Automotiva Imagens Leilão", "AUTO_IMAGENS_LEILAO",
            RestServiceConstants.Nexo.IMAGENS_LEILAO, TipoBilhete.NEXO_IMAGENS_LEILAO),
    NEXO_SENATRAN_VEICULAR("Automotiva Senatran Veicular", "AUTO_SENATRAN_VEICULAR",
            RestServiceConstants.Nexo.SENATRAN_VEICULAR, TipoBilhete.NEXO_SENATRAN_VEICULAR),
    NEXO_CRLV_SIMPLES("Automotiva Crlv Simples", "AUTO_CRLV_SIMPLES",
            RestServiceConstants.Nexo.CRLV_SIMPLES, TipoBilhete.NEXO_CRLV),

    ULTRACHECK_QUILOMETRAGEM("Automotiva Quilometragem2", "AUTO_QUILOMETRAGEM",
            RestServiceConstants.Unicheck.QUILOMETRAGEM, TipoBilhete.ULTRACHECK_QUILOMETRAGEM),
    ULTRACHECK_ALERTA_VEICULAR("Alerta Veicular", "AUTO_ALERTA_VEICULAR",
            RestServiceConstants.Unicheck.ALERTA_VEICULAR, TipoBilhete.ULTRACHECK_ALERTA_VEICULAR),
    ULTRACHECK_LEILAO_BASE01_SCORE_VEICULAR("Automotiva Leilão Base 01 Score Veicular", "AUTO_LEILAO_BASE01_SCORE_VEICULAR",
            RestServiceConstants.Unicheck.LEILAO_BASE01_SCORE_VEICULAR, TipoBilhete.ULTRACHECK_LEILAO_BASE01_SCORE_VEICULAR),

    ULTRACHECK_RECALL("Automotiva Recall", "AUTO_RECALL",
            RestServiceConstants.Unicheck.RECALL, TipoBilhete.ULTRACHECK_RECALL),

    ULTRACHECK_INDICIO_DE_SINISTRO01("Automotiva Indício de Sinistro 01", "AUTO_INDICIO_DE_SINISTRO01",
            RestServiceConstants.Unicheck.INDICIO_DE_SINISTRO01, TipoBilhete.ULTRACHECK_INDICIO_DE_SINISTRO01),

    ULTRACHECK_ALERTA_DE_INDICIO("Automotiva Alerta de Indício", "AUTO_ALERTA_DE_INDICIO",
            RestServiceConstants.Unicheck.ALERTA_DE_INDICIO, TipoBilhete.ULTRACHECK_ALERTA_DE_INDICIO),

    ULTRACHECK_LOCALIZACAO_SIMPLES("Cadastral Localização Simples", "CADASTRAL_LOCALIZACAO_SIMPLES",
            RestServiceConstants.Unicheck.LOCALIZACAO_SIMPLES, TipoBilhete.ULTRACHECK_LOCALIZACAO_SIMPLES),

    ULTRACHECK_CHEQUE_CERTO("Cadastral Cheque Certo", "CADASTRAL_CHEQUE_CERTO",
            RestServiceConstants.Unicheck.CHEQUE_CERTO, TipoBilhete.ULTRACHECK_CHEQUE_CERTO),

    ULTRACHECK_CERTIDAO_NASCIONAL_DE_DEBITOS_TRAB("Automotiva Certidão Nascional de Débitos Trabalhistas", "AUTO_CERTIDAO_NASCIONAL_DE_DEBITOS_TRAB",
            RestServiceConstants.Unicheck.ULTRACHECK_CERTIDAO_NASCIONAL_DE_DEBITOS_TRAB, TipoBilhete.ULTRACHECK_CERTIDAO_NASCIONAL_DE_DEBITOS_TRAB),

    ULTRACHECK_SRC_BACEN_SCORE_POSITIVO("Cadastral SRC Bacen Score Positivo", "CADASTRAL_SRC_BACEN_SCORE_POSITIVO",
            RestServiceConstants.Unicheck.SRC_BACEN_SCORE_POSITIVO, TipoBilhete.ULTRACHECK_SRC_BACEN_SCORE_POSITIVO),

    ULTRACHECK_PROTESTO_CENPROT("Cadastral Protesto CENPROT", "CADASTRAL_PROTESTO_CENPROT",
            RestServiceConstants.Unicheck.PROTESTO_CENPROT, TipoBilhete.ULTRACHECK_PROTESTO_CENPROT),

    ULTRACHECK_CHEQUE_CONTUMACIA("Cadastral Cheque Contumácia", "CADASTRAL_CHEQUE_CONTUMACIA",
            RestServiceConstants.Unicheck.CHEQUE_CONTUMACIA, TipoBilhete.ULTRACHECK_CHEQUE_CONTUMACIA),

    LEILAO_LOCAL("Automotiva Leilao Local", "AUTO_LEILAO_LOCAL",
            RestServiceConstants.Unicheck.LEILAO_LOCAL, TipoBilhete.ULTRACHECK_LEILAO_LOCAL),

    SINISTRO_LOCAL("Automotiva Tabela Consulta Sinistro", "AUTO_SINISTRO_LOCAL",
            RestServiceConstants.Unicheck.TABELA_CONSULTA, TipoBilhete.SINISTRO_TABELA_CONSULTA),

    ULTRACHECK_AUTO_PRECIFICACAO("Automotiva Consulta de Precificação", "AUTO_PRECIFICACAO",
            RestServiceConstants.Unicheck.AUTO_PRECIFICACAO, TipoBilhete.SINISTRO_TABELA_CONSULTA),

    ULTRACHECK_FICHA_TECNICA("Automotiva Consulta Ficha Técnica", "AUTO_FICHA_TECNICA",
            RestServiceConstants.Unicheck.FICHA_TECNICA, TipoBilhete.ULTRACHECK_FICHA_TECNICA);

    OpcaoConsulta(String descricao, String permissao, String resposta, TipoBilhete tipoBilhete) {
        this.resposta = resposta;
        this.descricao = descricao;
        this.permissao = permissao;
        this.tipoBilhete = tipoBilhete;
    }

    private final String descricao;
    private final String resposta;
    private final String permissao;
    private final TipoBilhete tipoBilhete;

    public TipoBilhete getTipoBilhete() {
        return tipoBilhete;
    }

    public String getResposta() {
        return resposta;
    }

    public String getDescricao() {
        return descricao;
    }

    public String getPermissao() {
        return permissao;
    }

    public static OpcaoConsulta getByPermissao(String permissao) {
        return Arrays.stream(OpcaoConsulta.values()).filter(el -> el.permissao.equals(permissao)).findFirst().orElse(null);
    }

}
