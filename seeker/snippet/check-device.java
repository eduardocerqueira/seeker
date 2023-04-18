//date: 2023-04-18T16:54:15Z
//url: https://api.github.com/gists/c74f077d3b501e31587c3b7c93b0ca5a
//owner: https://api.github.com/users/lakshya-aggarwal

package com.example.models;

import org.apache.sling.api.SlingHttpServletRequest;
import org.apache.sling.models.annotations.DefaultInjectionStrategy;
import org.apache.sling.models.annotations.Model;
import org.apache.sling.models.annotations.injectorspecific.OSGiService;

import com.day.cq.wcm.api.designer.Designer;
import com.day.cq.wcm.api.designer.Designer.Style;
import com.day.cq.wcm.foundation.Paragraph;

@Model(adaptables = SlingHttpServletRequest.class, defaultInjectionStrategy = DefaultInjectionStrategy.OPTIONAL)
public class DeviceModel {

    @OSGiService
    private Designer designer;

    private boolean isMobile;
    private boolean isTablet;
    private boolean isDesktop;

    public boolean isMobile() {
        return isMobile;
    }

    public boolean isTablet() {
        return isTablet;
    }

    public boolean isDesktop() {
        return isDesktop;
    }

    public void init() {
        SlingHttpServletRequest request = (SlingHttpServletRequest) getCurrentResource().getResourceResolver().adaptTo(SlingHttpServletRequest.class);
        if (request != null) {
            Style style = designer.getStyle(request);
            if (style != null) {
                Paragraph paragraph = new Paragraph(style);
                isMobile = paragraph.isMobile();
                isTablet = paragraph.isTablet();
                isDesktop = paragraph.isDesktop();
            }
        }
    }
}
