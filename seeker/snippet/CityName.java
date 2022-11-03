//date: 2022-11-03T17:06:15Z
//url: https://api.github.com/gists/566fad367f77c08be4e7985cfd9b3569
//owner: https://api.github.com/users/kodefyi

package com.kodefyi.cityproject.entity;

import java.util.Comparator;

public class CityName implements Comparator<City> {

	@Override
	public int compare(City o1, City o2) {
		// TODO Auto-generated method stub
		return o1.getCityName().compareTo(o2.getCityName());
	}

}