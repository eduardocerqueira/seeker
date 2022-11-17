//date: 2022-11-17T17:12:54Z
//url: https://api.github.com/gists/97b352a423d615b62e2044e33e779652
//owner: https://api.github.com/users/Ana-Chequer

package br.com.ana.cadastrorevistas.model;

import java.util.HashSet;
import java.util.Set;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.JoinTable;
import javax.persistence.ManyToMany;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@Entity
public class Tag {
	
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long id;
	
	@Column(nullable = false)
	private String nome;
	
	@JoinTable(name = "tag_revista", 
			joinColumns = @JoinColumn(name = "tag_id"),
			inverseJoinColumns = @JoinColumn(name = "revista_id"))
	
	@ManyToMany(cascade = {CascadeType.PERSIST, CascadeType.MERGE})
	private Set<Revista> revistas = new HashSet<>();

	public Tag(String nome) {
		this.nome = nome;
	}
		
	public void adicionar(Revista revista) {
		this.revistas.add(revista);
		revista.adicionar(this);
	}
	

}
