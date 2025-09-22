//date: 2025-09-22T17:01:14Z
//url: https://api.github.com/gists/c794631992d49a38565d141967b27c6a
//owner: https://api.github.com/users/srinuyadav997

package controller;

import java.io.IOException;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import dao.HospitalDao;
import dto.Hospital;
import service.HospitalService;
@WebServlet("/edit")
public class EditHospital extends HttpServlet {
HospitalService hospitalService=new HospitalService();
@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		int id=Integer.parseInt(req.getParameter("id"));
		Hospital hospital=hospitalService.getHospitalServiceById(id);
		if (hospital!=null) {
			RequestDispatcher dispatcher=req.getRequestDispatcher("updatehospital.jsp");
			dispatcher.forward(req, resp);
		}
		else {
			RequestDispatcher dispatcher=req.getRequestDispatcher("edithospital.jsp");
			dispatcher.forward(req, resp);
		}
		
	}
}
