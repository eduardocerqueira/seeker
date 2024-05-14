//date: 2024-05-14T17:02:09Z
//url: https://api.github.com/gists/b560bcf1f3c25c35548d11432669460b
//owner: https://api.github.com/users/groupdocs-com-kb

import com.groupdocs.metadata.Metadata;
import com.groupdocs.metadata.core.FileFormat;
import com.groupdocs.metadata.core.IReadOnlyList;
import com.groupdocs.metadata.core.MetadataProperty;
import com.groupdocs.metadata.core.MetadataPropertyType;
import com.groupdocs.metadata.licensing.License;
import com.groupdocs.metadata.search.FallsIntoCategorySpecification;
import com.groupdocs.metadata.search.OfTypeSpecification;
import com.groupdocs.metadata.search.Specification;
import com.groupdocs.metadata.tagging.Tags;
import java.util.Calendar;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ReadMetadataFromPPTXUsingJava {
        public static void main(String[] args) {

            // Set License to avoid the limitations of Metadata library
            License license = new License();
            license.setLicense("GroupDocs.Metadata.lic");

            Metadata metadata = new Metadata("input.docx");
            if (metadata.getFileFormat() != FileFormat.Unknown && !metadata.getDocumentInfo().isEncrypted()) {
                System.out.println();

                // Fetch all metadata properties that fall into a particular category
                IReadOnlyList<MetadataProperty> properties = metadata.findProperties(new FallsIntoCategorySpecification(Tags.getContent()));
                System.out.println("The metadata properties describing some characteristics of the file content: title, keywords, language, etc.");
                for (MetadataProperty property : properties) {
                    System.out.println(String.format("Property name: %s, Property value: %s", property.getName(), property.getValue()));
                }

                // Fetch all properties having a specific type and value
                int year = Calendar.getInstance().get(Calendar.YEAR);
                properties = metadata.findProperties(new OfTypeSpecification(MetadataPropertyType.DateTime).and(new ReadMetadataFromPPTXUsingJava().new YearMatchSpecification(year)));
                System.out.println("All datetime properties with the year value equal to the current year");
                for (MetadataProperty property : properties) {
                    System.out.println(String.format("Property name: %s, Property value: %s", property.getName(), property.getValue()));
                }

                // Fetch all properties whose names match the specified regex
                Pattern pattern = Pattern.compile("^author|company|(.+date.*)$", Pattern.CASE_INSENSITIVE);
                properties = metadata.findProperties(new ReadMetadataFromPPTXUsingJava().new RegexSpecification(pattern));
                System.out.println(String.format("All properties whose names match the following regex: %s", pattern.pattern()));
                for (MetadataProperty property : properties) {
                    System.out.println(String.format("Property name: %s, Property value: %s", property.getName(), property.getValue()));
                }
           }
    }

    // Define your own specifications to filter metadata properties
    public class YearMatchSpecification extends Specification {
        public YearMatchSpecification(int year) {
            setValue(year);
        }

        public final int getValue() {
            return auto_Value;
        }

        private void setValue(int value) {
            auto_Value = value;
        }

        private int auto_Value;

        public boolean isSatisfiedBy(MetadataProperty candidate) {
            Date date = candidate.getValue().toClass(Date.class);
            if (date != null) {
                Calendar calendar = Calendar.getInstance();
                calendar.setTime(date);
                return getValue() == calendar.get(Calendar.YEAR);
            }
            return false;
        }
    }

    public class RegexSpecification extends Specification {
            private Pattern pattern;
            public RegexSpecification(Pattern pattern) {
                this.pattern = pattern;
            }

            @Override
            public boolean isSatisfiedBy(MetadataProperty metadataProperty) {
                Matcher matcher = pattern.matcher(metadataProperty.getName());
                return matcher.find();
            }
    }
}
