//date: 2022-01-12T17:16:22Z
//url: https://api.github.com/gists/2fbb04235b2ccd379f513a3a0f39cb1d
//owner: https://api.github.com/users/walteralleyz

interface Desconto {
 
    BigDecimal calcular(Pedido pedido);
 
    void setProximo(Desconto proximo);
}

class DescontoPorItens implements Desconto {
        
    private Desconto proximo;
 
    @Override
    public BigDecimal calcular(Pedido pedido) {
        if(pedido.getItens().size() > 10) {
            return pedido.getValor().multiply(BigDecimal.valueOf(0.5));
        } else {
            return proximo.calcular(pedido);
       }
    }
 
    @Override
    public void setProximo(Desconto proximo) {
       this.proximo = proximo;
    }
}

class DescontoPorValor implements Desconto {
        
     private Desconto proximo;
 
     @Override
     public BigDecimal calcular(Pedido pedido) {
        if(pedido.getValor() > 1000.0) {
            return pedido.getValor().multiply(BigDecimal.valueOf(0.10));
        } else {
            return proximo.calcular(pedido);
       }
     }
 
     @Override
     public void setProximo(Desconto proximo) {
         this.proximo = proximo;
     }
}

class SemDesconto implements Desconto {
     
    @Override
    public BigDecimal calcular(Pedido pedido) {
       return BigDecimal.ZERO;
    }
 
    @Override
    public void setProximo(Desconto desconto) {
       // Não faz nada, é o último nível
    }
}

class CalculadorDeDesconto {
 
   public static void main(String[] args) {
       final Pedido pedido = new Pedido;
       calculaDesconto(pedido);
   }   
 
   public BigDecimal calculaDesconto(Pedido pedido) {
        final Desconto descontoPorItem = new DescontoPorItens();
        final Desconto descontoPorValor = new DescontoPorValor();
        final Desconto semDesconto = new SemDesconto();
 
        descontoPorItem.setProximo(descontoPorValor);
        descontoPorValor.setProximo(semDesconto);
 
        return descontoPorItem.calcular(pedido);
   }
}