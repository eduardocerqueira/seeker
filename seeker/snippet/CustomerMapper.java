//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv.impl;

import me.pacphi.ai.resos.csv.CsvEntityMapper;
import me.pacphi.ai.resos.csv.CsvMappingException;
import me.pacphi.ai.resos.csv.EntityMapper;
import me.pacphi.ai.resos.jdbc.CustomerEntity;

import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;

@CsvEntityMapper("customers")
public class CustomerMapper implements EntityMapper<CustomerEntity> {

    @Override
    public CustomerEntity mapFromCsv(String[] line) throws CsvMappingException {
        try {
            var entity = new CustomerEntity();
            entity.setName(line[0]);
            entity.setEmail(line[1]);
            entity.setPhone(line[2]);
            entity.setCreatedAt(OffsetDateTime.parse(line[3]));
            entity.setLastBookingAt(OffsetDateTime.parse(line[4]));
            entity.setBookingCount(Integer.parseInt(line[5]));
            entity.setTotalSpent(Float.parseFloat(line[6]));
            entity.setMetadata(line[7]);
            return entity;
        } catch (DateTimeParseException | IllegalArgumentException | NullPointerException e) {
            throw new CsvMappingException("Failed to map customer from CSV", e);
        }
    }

    @Override
    public Class<CustomerEntity> getEntityClass() {
        return CustomerEntity.class;
    }
}
