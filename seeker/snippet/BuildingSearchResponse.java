//date: 2022-06-06T16:54:05Z
//url: https://api.github.com/gists/d39becf1bb925c910d45dab68edf30f3
//owner: https://api.github.com/users/patricktran9

package com.tutorial.model.response;

public class BuildingSearchResponse {
	private String name;
	private String address;
	private String managerName;
	private String managerPhone;
	private Integer floorArea;
	private String rentAreaValue;
	private Integer rentPrice;
	private String serviceFee;
	private Float brokerageFee;
	private String createdDate;
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getAddress() {
		return address;
	}
	public void setAddress(String address) {
		this.address = address;
	}
	public String getManagerName() {
		return managerName;
	}
	public void setManagerName(String managerName) {
		this.managerName = managerName;
	}
	public String getManagerPhone() {
		return managerPhone;
	}
	public void setManagerPhone(String managerPhone) {
		this.managerPhone = managerPhone;
	}
	public Integer getFloorArea() {
		return floorArea;
	}
	public void setFloorArea(Integer floorArea) {
		this.floorArea = floorArea;
	}
	public String getRentAreaValue() {
		return rentAreaValue;
	}
	public void setRentAreaValue(String rentAreaValue) {
		this.rentAreaValue = rentAreaValue;
	}
	public Integer getRentPrice() {
		return rentPrice;
	}
	public void setRentPrice(Integer rentPrice) {
		this.rentPrice = rentPrice;
	}
	public String getServiceFee() {
		return serviceFee;
	}
	public void setServiceFee(String serviceFee) {
		this.serviceFee = serviceFee;
	}
	public Float getBrokerageFee() {
		return brokerageFee;
	}
	public void setBrokerageFee(Float brokerageFee) {
		this.brokerageFee = brokerageFee;
	}
	public String getCreatedDate() {
		return createdDate;
	}
	public void setCreatedDate(String createdDate) {
		this.createdDate = createdDate;
	}
	
}