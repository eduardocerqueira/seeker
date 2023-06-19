//date: 2023-06-19T16:57:49Z
//url: https://api.github.com/gists/f698d03e8c99b3426d230a70c5308261
//owner: https://api.github.com/users/khuonghung

package com.hnt.dental.service;

import com.hnt.dental.util.SecretUtls;
import jakarta.servlet.http.HttpServletRequest;
import org.apache.commons.lang3.StringUtils;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.util.*;

public class VNPayService {

    private static final ResourceBundle bundle = ResourceBundle.getBundle("application");

    public String renderPayment(Long bookingId, String txnRef, int amount, HttpServletRequest request) {
        String vnpVersion = "2.0.0";
        String vnpCommand = "pay";

        String orderType = "billpayment";
        String vnpIpAddr = getIpAddress(request);
        String vnpTmnCode = bundle.getString("vnp.tmnCode");
        Map<String, String> vnpParams = new HashMap<>();
        vnpParams.put("vnp_Version", vnpVersion);
        vnpParams.put("vnp_Command", vnpCommand);
        vnpParams.put("vnp_TmnCode", vnpTmnCode);
        vnpParams.put("vnp_Amount", String.valueOf(amount));
        vnpParams.put("vnp_CurrCode", "VND");
        vnpParams.put("vnp_BankCode", "");
        vnpParams.put("vnp_TxnRef", bundle.getString("vnp.tmnCode") + txnRef);
        vnpParams.put("vnp_OrderInfo", StringUtils.join("bookingId: ", bookingId));
        vnpParams.put("vnp_OrderType", orderType);

        String locate = "vi";
        vnpParams.put("vnp_Locale", locate);
        vnpParams.put("vnp_ReturnUrl", bundle.getString("vnp.return.url"));
        vnpParams.put("vnp_IpAddr", vnpIpAddr);

        Date now = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyyMMddHHmmss");
        String vnpCreateDate = formatter.format(now);
        vnpParams.put("vnp_CreateDate", vnpCreateDate);

        List<String> fieldNames = new ArrayList<>(vnpParams.keySet());
        Collections.sort(fieldNames);
        StringBuilder hashData = new StringBuilder();
        StringBuilder query = new StringBuilder();
        Iterator<String> itr = fieldNames.iterator();
        while (itr.hasNext()) {
            String fieldName = itr.next();
            String fieldValue = vnpParams.get(fieldName);
            if ((fieldValue != null) && (fieldValue.length() > 0)) {
                hashData.append(fieldName);
                hashData.append('=');
                hashData.append(fieldValue);
                query.append(URLEncoder.encode(fieldName, StandardCharsets.US_ASCII));
                query.append('=');
                query.append(URLEncoder.encode(fieldValue, StandardCharsets.US_ASCII));
                if (itr.hasNext()) {
                    query.append('&');
                    hashData.append('&');
                }
            }
        }
        String queryUrl = query.toString();
        String vnpSecureHash = "**********"
        queryUrl += "&vnp_SecureHashType=SHA256&vnp_SecureHash=" + vnpSecureHash;
        return bundle.getString("vnp.pay.url") + "?" + queryUrl;
    }

    public static String getIpAddress(HttpServletRequest request) {
        String ipAdress;
        try {
            ipAdress = request.getHeader("X-FORWARDED-FOR");
            if (ipAdress == null) {
                ipAdress = request.getRemoteAddr();
            }
        } catch (Exception e) {
            ipAdress = "Invalid IP:" + e.getMessage();
        }
        return ipAdress;
    }
}
Message();
        }
        return ipAdress;
    }
}
