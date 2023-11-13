//date: 2023-11-13T16:52:40Z
//url: https://api.github.com/gists/2c1688251b6ac0bda1b1d03776ad0a12
//owner: https://api.github.com/users/JadenFurtado

package in.org.npci.upiapp.nfc;

import android.content.Intent;
import android.nfc.cardemulation.HostApduService;
import android.os.Bundle;
import android.util.Pair;
import com.google.gson.C4826e;
import fjv.C5737b;
import in.org.npci.upiapp.core.NPCIJSInterface;
import in.org.npci.upiapp.nfc.art.ARTPayload;
import in.org.npci.upiapp.nfc.art.ErrorContext;
import in.org.npci.upiapp.nfc.models.AssertIdentity;
import in.org.npci.upiapp.nfc.models.AuthData;
import in.org.npci.upiapp.nfc.models.ContextData;
import in.org.npci.upiapp.nfc.models.HandshakeModel;
import in.org.npci.upiapp.nfc.models.LegRequest;
import in.org.npci.upiapp.nfc.models.LegResponse;
import in.org.npci.upiapp.nfc.models.PayeeInfo;
import in.org.npci.upiapp.nfc.models.PurseInfo;
import in.org.npci.upiapp.nfc.models.StartCredit;
import in.org.npci.upiapp.nfc.models.StartTransferData;
import in.org.npci.upiapp.nfc.models.TransferValue;
import in.org.npci.upiapp.nfc.models.TxnDetails;
import in.org.npci.upiapp.nfc.models.VerifyIdentity;
import in.org.npci.upiapp.nfc.models.VtpResponse;
import in.org.npci.upiapp.nfc.utils.NfcConstants;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;
import org.npci.upi.security.services.CLServices;
import p285s0.C8218a;
import p309te.C8563b;
import ve.C8929a;
/* loaded from: classes2.dex */
public class PayeeHostingService extends HostApduService {

    /* renamed from: A */
    String f17567A;

    /* renamed from: c */
    CLServices f17578c;

    /* renamed from: d */
    C8929a f17579d;

    /* renamed from: q */
    byte f17582q;

    /* renamed from: s */
    PayeeInfo f17584s;

    /* renamed from: t */
    PurseInfo f17585t;

    /* renamed from: u */
    ContextData f17586u;

    /* renamed from: v */
    ContextData f17587v;

    /* renamed from: w */
    AssertIdentity f17588w;

    /* renamed from: x */
    VerifyIdentity f17589x;

    /* renamed from: y */
    TxnDetails f17590y;

    /* renamed from: z */
    StartCredit f17591z;

    /* renamed from: a */
    String f17576a = PayeeHostingService.class.getName();

    /* renamed from: b */
    boolean f17577b = false;

    /* renamed from: e */
    int f17580e = 0;

    /* renamed from: p */
    byte f17581p = 0;

    /* renamed from: r */
    int f17583r = 0;

    /* renamed from: B */
    public String f17568B = C5737b.m11631a(6724);

    /* renamed from: C */
    public String f17569C = C5737b.m11631a(6725);

    /* renamed from: D */
    public String f17570D = C5737b.m11631a(6726);

    /* renamed from: E */
    public String f17571E = C5737b.m11631a(6727);

    /* renamed from: F */
    public String f17572F = C5737b.m11631a(6728);

    /* renamed from: G */
    public String f17573G = C5737b.m11631a(6729);

    /* renamed from: H */
    public String f17574H = C5737b.m11631a(6730);

    /* renamed from: I */
    public String f17575I = C5737b.m11631a(6731);

    /* renamed from: in.org.npci.upiapp.nfc.PayeeHostingService$a */
    /* loaded from: classes2.dex */
    class RunnableC6370a implements Runnable {

        /* renamed from: a */
        final /* synthetic */ String f17592a;

        /* renamed from: b */
        final /* synthetic */ String[] f17593b;

        RunnableC6370a(String str, String[] strArr) {
            this.f17592a = str;
            this.f17593b = strArr;
        }

        @Override // java.lang.Runnable
        public void run() {
            try {
                if (this.f17592a.equalsIgnoreCase(C5737b.m11631a(7665))) {
                    PayeeHostingService.this.sendResponseApdu(null);
                } else if (NfcConstants.m9805m(this.f17592a)) {
                    PayeeHostingService payeeHostingService = PayeeHostingService.this;
                    payeeHostingService.m9822c(payeeHostingService.f17568B, payeeHostingService.f17574H);
                    byte[] bytes = NfcConstants.m9812f(C5737b.m11631a(7666)).getBytes();
                    PayeeHostingService payeeHostingService2 = PayeeHostingService.this;
                    PayeeHostingService.this.sendResponseApdu(NfcConstants.m9811g(bytes, payeeHostingService2.f17581p, payeeHostingService2.f17582q, (byte) 1));
                } else if (NfcConstants.m9813e(this.f17592a)) {
                    PayeeHostingService payeeHostingService3 = PayeeHostingService.this;
                    payeeHostingService3.m9822c(payeeHostingService3.f17568B, payeeHostingService3.f17575I);
                } else {
                    this.f17593b[0] = PayeeHostingService.this.m9824a(this.f17592a);
                    PayeeHostingService payeeHostingService4 = PayeeHostingService.this;
                    if (payeeHostingService4.f17580e == 5) {
                        payeeHostingService4.m9819f(new C4826e().m13605r(PayeeHostingService.this.f17590y));
                    }
                    PayeeHostingService payeeHostingService5 = PayeeHostingService.this;
                    payeeHostingService5.f17580e++;
                    byte[] bytes2 = this.f17593b[0].getBytes();
                    PayeeHostingService payeeHostingService6 = PayeeHostingService.this;
                    payeeHostingService5.sendResponseApdu(NfcConstants.m9811g(bytes2, payeeHostingService6.f17581p, payeeHostingService6.f17582q, (byte) 1));
                }
            } catch (Exception unused) {
                PayeeHostingService payeeHostingService7 = PayeeHostingService.this;
                payeeHostingService7.m9822c(payeeHostingService7.f17568B, PayeeHostingService.this.f17573G + C5737b.m11631a(7667) + PayeeHostingService.this.f17580e);
                byte[] bytes3 = NfcConstants.m9812f(C5737b.m11631a(7668)).getBytes();
                PayeeHostingService payeeHostingService8 = PayeeHostingService.this;
                PayeeHostingService.this.sendResponseApdu(NfcConstants.m9811g(bytes3, payeeHostingService8.f17581p, payeeHostingService8.f17582q, (byte) 1));
            }
        }
    }

    /* renamed from: b */
    private byte[] m9823b() {
        return NfcConstants.f17602d.getBytes();
    }

    /* renamed from: g */
    private boolean m9818g(byte[] bArr) {
        if (bArr.length >= 2 && bArr[0] == 0 && bArr[1] == -92) {
            return true;
        }
        return false;
    }

    /* renamed from: a */
    public String m9824a(String str) {
        String str2;
        LegResponse legResponse;
        String m11631a = C5737b.m11631a(6732);
        String m11631a2 = C5737b.m11631a(6733);
        String m11631a3 = C5737b.m11631a(6734);
        String m11631a4 = C5737b.m11631a(6735);
        String m11631a5 = C5737b.m11631a(6736);
        String m11631a6 = C5737b.m11631a(6737);
        String m11631a7 = C5737b.m11631a(6738);
        String m11631a8 = C5737b.m11631a(6739);
        String m11631a9 = C5737b.m11631a(6740);
        try {
            int i = this.f17580e;
            String m11631a10 = C5737b.m11631a(6741);
            try {
                if (i != 1) {
                    if (i != 2) {
                        if (i != 3) {
                            if (i != 4) {
                                if (i == 5) {
                                    TxnDetails txnDetails = new TxnDetails();
                                    this.f17590y = txnDetails;
                                    txnDetails.setTxnId(this.f17591z.getTxnId());
                                    this.f17590y.setPayerVPA(NfcConstants.m9808j(this.f17586u.getPayerInfo()));
                                    this.f17590y.setPayeeVPA(this.f17584s.getPayeeVPA());
                                    this.f17590y.setPayeeQR(this.f17587v.getPayeeInfo());
                                    this.f17590y.setPayerQR(this.f17586u.getPayerInfo());
                                    this.f17590y.setStatus(C5737b.m11631a(6742));
                                    this.f17590y.setAmount(this.f17586u.getAmount());
                                    this.f17590y.setDateTime(this.f17567A);
                                    this.f17590y.setPayerTxn(false);
                                    if (((LegRequest) new C4826e().m13615h(((ARTPayload) new C4826e().m13615h(str, ARTPayload.class)).getData(), LegRequest.class)).getVtpRequest().getType().equalsIgnoreCase(m11631a)) {
                                        LegResponse legResponse2 = new LegResponse();
                                        VtpResponse vtpResponse = new VtpResponse();
                                        vtpResponse.setStatus(m11631a10);
                                        vtpResponse.setType(m11631a);
                                        vtpResponse.setData(NfcConstants.m9809i(new JSONObject()));
                                        legResponse2.setVtp_response(vtpResponse);
                                        legResponse = legResponse2;
                                    }
                                }
                                legResponse = null;
                            } else {
                                ARTPayload aRTPayload = (ARTPayload) new C4826e().m13615h(str, ARTPayload.class);
                                if (aRTPayload.isSuccess()) {
                                    LegRequest legRequest = (LegRequest) new C4826e().m13615h(aRTPayload.getData(), LegRequest.class);
                                    if (legRequest.getVtpRequest().getType().equalsIgnoreCase(m11631a5)) {
                                        Pair<String, String> m2405k = this.f17579d.m2405k(this.f17584s.getMobile(), this.f17584s.getDeviceId(), this.f17584s.getReferenceId(), ((TransferValue) new C4826e().m13615h(new JSONObject(legRequest.getVtpRequest().getData().toString()).getString(C5737b.m11631a(6743)), TransferValue.class)).getSignature());
                                        if (((String) m2405k.first).equals(m11631a10)) {
                                            TransferValue transferValue = (TransferValue) new C4826e().m13615h((String) m2405k.second, TransferValue.class);
                                            legResponse = new LegResponse();
                                            VtpResponse vtpResponse2 = new VtpResponse();
                                            vtpResponse2.setStatus(m11631a10);
                                            vtpResponse2.setType(m11631a5);
                                            JSONObject jSONObject = new JSONObject();
                                            jSONObject.put(C5737b.m11631a(6744), new JSONObject((String) m2405k.second));
                                            vtpResponse2.setData(NfcConstants.m9809i(jSONObject));
                                            legResponse.setVtp_response(vtpResponse2);
                                        } else {
                                            this.f17583r = this.f17580e;
                                        }
                                    }
                                }
                                legResponse = null;
                            }
                        } else {
                            ARTPayload aRTPayload2 = (ARTPayload) new C4826e().m13615h(str, ARTPayload.class);
                            if (aRTPayload2.isSuccess()) {
                                LegRequest legRequest2 = (LegRequest) new C4826e().m13615h(aRTPayload2.getData(), LegRequest.class);
                                if (legRequest2.getVtpRequest().getType().equalsIgnoreCase(m11631a6)) {
                                    StartTransferData startTransferData = (StartTransferData) new C4826e().m13615h(new JSONObject(legRequest2.getVtpRequest().getData().toString()).getString(C5737b.m11631a(6745)), StartTransferData.class);
                                    this.f17567A = startTransferData.getTimestamp();
                                    Pair<String, String> m2407i = this.f17579d.m2407i(this.f17584s.getMobile(), this.f17584s.getDeviceId(), this.f17584s.getReferenceId(), startTransferData.getTimestamp(), startTransferData.getAmount(), this.f17586u, this.f17587v);
                                    if (((String) m2407i.first).equals(m11631a10)) {
                                        this.f17591z = (StartCredit) new C4826e().m13615h((String) m2407i.second, StartCredit.class);
                                        legResponse = new LegResponse();
                                        VtpResponse vtpResponse3 = new VtpResponse();
                                        vtpResponse3.setStatus(m11631a10);
                                        vtpResponse3.setType(m11631a6);
                                        JSONObject jSONObject2 = new JSONObject();
                                        jSONObject2.put(C5737b.m11631a(6746), new JSONObject((String) m2407i.second));
                                        vtpResponse3.setData(NfcConstants.m9809i(jSONObject2));
                                        legResponse.setVtp_response(vtpResponse3);
                                    } else {
                                        this.f17583r = this.f17580e;
                                    }
                                }
                            }
                            legResponse = null;
                        }
                    } else {
                        ARTPayload aRTPayload3 = (ARTPayload) new C4826e().m13615h(str, ARTPayload.class);
                        if (aRTPayload3.isSuccess()) {
                            LegRequest legRequest3 = (LegRequest) new C4826e().m13615h(aRTPayload3.getData(), LegRequest.class);
                            if (legRequest3.getVtpRequest().getType().equalsIgnoreCase(m11631a7)) {
                                Pair<String, String> m2404l = this.f17579d.m2404l(this.f17584s.getMobile(), this.f17584s.getDeviceId(), this.f17584s.getReferenceId(), this.f17586u, (AuthData) new C4826e().m13615h(new JSONObject(legRequest3.getVtpRequest().getData().toString()).getString(m11631a3), AuthData.class));
                                if (((String) m2404l.first).equals(m11631a10)) {
                                    this.f17589x = (VerifyIdentity) new C4826e().m13615h((String) m2404l.second, VerifyIdentity.class);
                                    Pair<String, String> m2414b = this.f17579d.m2414b(this.f17584s.getMobile(), this.f17584s.getDeviceId(), this.f17584s.getReferenceId(), this.f17586u);
                                    if (((String) m2414b.first).equals(m11631a10)) {
                                        this.f17588w = (AssertIdentity) new C4826e().m13615h((String) m2414b.second, AssertIdentity.class);
                                        legResponse = new LegResponse();
                                        VtpResponse vtpResponse4 = new VtpResponse();
                                        vtpResponse4.setStatus(m11631a10);
                                        vtpResponse4.setType(m11631a7);
                                        JSONObject jSONObject3 = new JSONObject();
                                        jSONObject3.put(m11631a3, new JSONObject((String) m2414b.second));
                                        vtpResponse4.setData(NfcConstants.m9809i(jSONObject3));
                                        legResponse.setVtp_response(vtpResponse4);
                                    }
                                } else {
                                    this.f17583r = this.f17580e;
                                }
                            }
                        }
                        legResponse = null;
                    }
                } else {
                    ARTPayload aRTPayload4 = (ARTPayload) new C4826e().m13615h(str, ARTPayload.class);
                    if (aRTPayload4.isSuccess()) {
                        LegRequest legRequest4 = (LegRequest) new C4826e().m13615h(aRTPayload4.getData(), LegRequest.class);
                        if (legRequest4.getVtpRequest().getType().equalsIgnoreCase(m11631a8)) {
                            this.f17586u = (ContextData) new C4826e().m13615h(new JSONObject(legRequest4.getVtpRequest().getData().toString()).getString(m11631a4), ContextData.class);
                            if (this.f17584s == null) {
                                String m2412d = this.f17579d.m2412d();
                                PrintStream printStream = System.out;
                                printStream.println(m11631a2 + m2412d);
                                this.f17584s = (PayeeInfo) new C4826e().m13615h(m2412d, PayeeInfo.class);
                                PrintStream printStream2 = System.out;
                                printStream2.println(m11631a2 + this.f17584s);
                            }
                            Pair<String, String> m2411e = this.f17579d.m2411e(this.f17584s.getMobile(), this.f17584s.getDeviceId(), this.f17584s.getReferenceId());
                            if (((String) m2411e.first).equals(m11631a10)) {
                                legResponse = new LegResponse();
                                VtpResponse vtpResponse5 = new VtpResponse();
                                vtpResponse5.setStatus(m11631a10);
                                this.f17585t = (PurseInfo) new C4826e().m13615h((String) m2411e.second, PurseInfo.class);
                                JSONObject jSONObject4 = new JSONObject();
                                JSONObject jSONObject5 = new JSONObject();
                                jSONObject5.put(C5737b.m11631a(6747), this.f17585t.getId());
                                jSONObject5.put(C5737b.m11631a(6748), this.f17585t.getSeq());
                                jSONObject5.put(C5737b.m11631a(6749), this.f17585t.getState());
                                jSONObject5.put(C5737b.m11631a(6750), this.f17585t.getProduct());
                                jSONObject5.put(C5737b.m11631a(6751), this.f17585t.getCurrency());
                                jSONObject5.put(C5737b.m11631a(6752), this.f17585t.getCountry());
                                jSONObject5.put(C5737b.m11631a(6753), this.f17585t.getType());
                                jSONObject5.put(C5737b.m11631a(6754), this.f17585t.getMcc());
                                jSONObject5.put(C5737b.m11631a(6755), this.f17585t.getBalance());
                                jSONObject5.put(C5737b.m11631a(6756), this.f17585t.getChallenge());
                                jSONObject5.put(C5737b.m11631a(6757), C5737b.m11631a(6758));
                                jSONObject5.put(C5737b.m11631a(6759), C5737b.m11631a(6760));
                                jSONObject5.put(C5737b.m11631a(6761), this.f17584s.getPayeeInfo());
                                jSONObject5.put(C5737b.m11631a(6762), this.f17586u.getAmount());
                                this.f17587v = (ContextData) new C4826e().m13615h(jSONObject5.toString(), ContextData.class);
                                jSONObject4.put(m11631a4, jSONObject5);
                                vtpResponse5.setData(NfcConstants.m9809i(jSONObject4));
                                vtpResponse5.setType(m11631a8);
                                legResponse.setVtp_response(vtpResponse5);
                            } else {
                                this.f17583r = this.f17580e;
                            }
                        }
                    }
                    legResponse = null;
                }
                if (this.f17580e == 0) {
                    HandshakeModel handshakeModel = new HandshakeModel(new int[]{1}, 0);
                    this.f17582q = (byte) 5;
                    return new C4826e().m13605r(new ARTPayload(true, NfcConstants.m9803o(handshakeModel), null));
                } else if (legResponse != null) {
                    this.f17582q = (byte) 2;
                    return new C4826e().m13605r(new ARTPayload(true, NfcConstants.m9803o(legResponse), null));
                } else {
                    this.f17582q = (byte) 2;
                    str2 = m11631a9;
                    try {
                        ARTPayload aRTPayload5 = new ARTPayload(false, null, new ErrorContext(str2));
                        NfcConstants.m9804n(this, C5737b.m11631a(6763));
                        return NfcConstants.m9803o(aRTPayload5);
                    } catch (Exception unused) {
                        ARTPayload aRTPayload6 = new ARTPayload(false, null, new ErrorContext(str2));
                        NfcConstants.m9804n(this, C5737b.m11631a(6764));
                        return new C4826e().m13605r(aRTPayload6);
                    }
                }
            } catch (Exception unused2) {
                str2 = m11631a9;
            }
        } catch (Exception unused3) {
            str2 = m11631a9;
        }
    }

    /* renamed from: c */
    public void m9822c(String str, String str2) {
        C8563b.m3245b().m3246a(str, str2);
    }

    /* renamed from: d */
    public void m9821d() {
        Intent intent = new Intent();
        intent.setAction(C5737b.m11631a(6765));
        C8218a.m4299b(this).m4297d(intent);
        CLServices cLServices = NPCIJSInterface.f17465d;
        this.f17578c = cLServices;
        this.f17579d = new C8929a(cLServices, this, C5737b.m11631a(6766));
        this.f17584s = (PayeeInfo) new C4826e().m13615h(this.f17579d.m2412d(), PayeeInfo.class);
    }

    /* renamed from: e */
    public void m9820e() {
        Intent intent = new Intent();
        intent.setAction(C5737b.m11631a(6767));
        C8218a.m4299b(this).m4297d(intent);
    }

    /* renamed from: f */
    public void m9819f(String str) {
        m9822c(this.f17568B, this.f17570D);
        Intent intent = new Intent();
        intent.setAction(C5737b.m11631a(6768));
        intent.putExtra(C5737b.m11631a(6769), str);
        C8218a.m4299b(this).m4297d(intent);
    }

    @Override // android.nfc.cardemulation.HostApduService
    public void onDeactivated(int i) {
        this.f17577b = false;
        m9820e();
    }

    @Override // android.nfc.cardemulation.HostApduService
    public byte[] processCommandApdu(byte[] bArr, Bundle bundle) {
        byte[] bArr2 = NfcConstants.f17599a;
        String str = this.f17568B;
        m9822c(str, this.f17572F + C5737b.m11631a(6770) + this.f17580e);
        if (m9818g(bArr)) {
            m9822c(this.f17568B, this.f17569C);
            this.f17577b = true;
            m9821d();
            this.f17580e = 0;
            return m9823b();
        }
        Pair<byte[], byte[]> m9815c = NfcConstants.m9815c(bArr);
        this.f17581p = NfcConstants.m9814d(bArr);
        try {
            new Thread(new RunnableC6370a(new String((byte[]) m9815c.second, StandardCharsets.UTF_8), new String[1])).start();
        } catch (Exception unused) {
        }
        return null;
    }
}
