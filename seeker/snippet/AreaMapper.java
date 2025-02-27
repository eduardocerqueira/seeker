//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv.impl;

import me.pacphi.ai.resos.csv.CsvEntityMapper;
import me.pacphi.ai.resos.csv.CsvMappingException;
import me.pacphi.ai.resos.csv.EntityMapper;
import me.pacphi.ai.resos.jdbc.AreaEntity;

@CsvEntityMapper("areas")
public class AreaMapper implements EntityMapper<AreaEntity> {

    @Override
    public AreaEntity mapFromCsv(String[] line) throws CsvMappingException {
        try {
            var entity = new AreaEntity();
            entity.setName(line[0]);
            entity.setInternalNote(line[1]);
            entity.setBookingPriority(Integer.parseInt(line[2]));
            return entity;
        } catch (IllegalArgumentException | NullPointerException e) {
            throw new CsvMappingException("Failed to map area from CSV", e);
        }
    }

    @Override
    public Class<AreaEntity> getEntityClass() {
        return AreaEntity.class;
    }
}
