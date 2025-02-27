//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv;

import jakarta.annotation.PostConstruct;
import liquibase.Liquibase;
import liquibase.database.Database;
import liquibase.database.DatabaseFactory;
import liquibase.database.jvm.JdbcConnection;
import liquibase.resource.ClassLoaderResourceAccessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ListableBeanFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import javax.sql.DataSource;
import java.io.IOException;
import java.nio.file.Path;
import java.sql.Connection;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@Profile(value = { "dev", "seed" })
public class DataSeeder implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(DataSeeder.class);

    private final DataSource dataSource;
    private final RepositoryResolver repositoryResolver;
    private final CsvFileProcessor csvFileProcessor;
    private final CsvResourceLoader resourceLoader;
    private final ListableBeanFactory beanFactory;
    private Map<String, EntityMapper<?>> entityMappers;

    public DataSeeder(
            DataSource dataSource,
            RepositoryResolver repositoryResolver,
            CsvFileProcessor csvFileProcessor,
            CsvResourceLoader resourceLoader,
            ListableBeanFactory beanFactory
            ) {
        this.dataSource = dataSource;
        this.repositoryResolver = repositoryResolver;
        this.csvFileProcessor = csvFileProcessor;
        this.resourceLoader = resourceLoader;
        this.beanFactory = beanFactory;
    }

    @PostConstruct
    void initializeMappers() {
        // Initialize entityMappers using @CsvEntityMapper annotation
        this.entityMappers = beanFactory.getBeansWithAnnotation(CsvEntityMapper.class).values().stream()
                .filter(bean -> bean instanceof EntityMapper)
                .map(bean -> (EntityMapper<?>) bean)
                .collect(Collectors.toMap(
                        mapper -> mapper.getClass().getAnnotation(CsvEntityMapper.class).value(),
                        Function.identity()
                ));
    }

    @Override
    @Transactional
    public void run(String... args) {
        if (!checkLiquibaseStatus()) {
            log.info("Skipping data seeding - Liquibase not ready");
            return;
        }

        List<Path> resolvedPaths = resourceLoader.resolveDataFiles();
        if (resolvedPaths.isEmpty()) {
            log.info("No CSV files found for seeding");
            return;
        }

        seedDatabase(resolvedPaths);
    }

    private boolean checkLiquibaseStatus() {
        try (Connection connection = dataSource.getConnection()) {
            Database database = DatabaseFactory.getInstance()
                    .findCorrectDatabaseImplementation(new JdbcConnection(connection));
            return new Liquibase("db/changelog/db.changelog-master.yml",
                    new ClassLoaderResourceAccessor(), database)
                    .isSafeToRunUpdate();
        } catch (Exception e) {
            log.error("Failed to check Liquibase status", e);
            return false;
        }
    }

    private void seedDatabase(List<Path> paths) {
        paths.forEach(this::processCsvFile);
    }

    private void processCsvFile(Path path) {
        String filePrefix = getFilePrefix(path);

        EntityMapper<?> mapper = entityMappers.get(filePrefix);
        if (mapper == null) {
            log.warn("No mapper found for file prefix: {}", filePrefix);
            return;
        }

        try {
            seedEntities(path, mapper);
        } catch (Exception e) {
            log.error("Failed to seed data from file: {}", path, e);
        }
    }

    private String getFilePrefix(Path path) {
        String fileName = path.getFileName().toString();
        return fileName.substring(0, fileName.indexOf('.'));
    }

    private <T> void seedEntities(Path csvPath, EntityMapper<T> mapper) throws IOException {
        Class<T> entityClass = mapper.getEntityClass();
        CrudRepository<T, ?> repository = repositoryResolver.getRepositoryForEntity(entityClass);

        List<T> entities = csvFileProcessor.processCsvFile(csvPath, line -> {
            try {
                return mapper.mapFromCsv(line);
            } catch (CsvMappingException e) {
                log.error("Failed to map CSV line: {}", Arrays.toString(line), e);
                return null;
            }
        });

        if (!entities.isEmpty()) {
            repository.saveAll(entities);
            log.info("Successfully seeded {} entities of type {}", entities.size(), entityClass.getSimpleName());
        }
    }
}