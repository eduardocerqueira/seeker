//date: 2022-08-01T17:21:49Z
//url: https://api.github.com/gists/d64d2caccc050c7d540496dd8b626ae1
//owner: https://api.github.com/users/danielmotazup

public class ReservaRequest {

    @JsonFormat(pattern = "dd/MM/yyyy")
    private LocalDate checkin;

    @JsonFormat(pattern = "dd/MM/yyyy")
    private LocalDate checkout;


    public LocalDate getCheckin() {
        return checkin;
    }

    public LocalDate getCheckout() {
        return checkout;
    }


    public Reserva toModel(){
        return new Reserva(this.checkin,this.checkout);
    }





}