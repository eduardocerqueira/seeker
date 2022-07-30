//date: 2022-07-30T19:00:50Z
//url: https://api.github.com/gists/52b8ea6b175985bfb2dfea5138f47b96
//owner: https://api.github.com/users/danyllosoareszup

public class ZupperResponse {

    private String nome;
    private String cargo;
    private int tempoDeCasa;
    private List<KudosResponse> kudos;
    private List<HabilidadesResponse> habilidades;
    private List<CertificadosResponse> certificados;

    public ZupperResponse(Zupper zupper) {
        this.nome = zupper.getNome();
        this.cargo = zupper.getCargo();
        this.tempoDeCasa = zupper.tempoDeCasa();
        this.kudos = zupper.getKudosRecebidos().stream().map(KudosResponse::new).collect(Collectors.toList());
        this.habilidades = zupper.getHabilidades().stream().map(HabilidadesResponse::new).collect(Collectors.toList());
        this.certificados = zupper.getCertificados().stream().map(CertificadosResponse::new).collect(Collectors.toList());
    }

    @Deprecated
    public ZupperResponse() {
    }

    public String getNome() {
        return nome;
    }

    public String getCargo() {
        return cargo;
    }

    public int getTempoDeCasa() {
        return tempoDeCasa;
    }

    public List<KudosResponse> getKudos() {
        return kudos;
    }

    public List<HabilidadesResponse> getHabilidades() {
        return habilidades;
    }

    public List<CertificadosResponse> getCertificados() {
        return certificados;
    }
}