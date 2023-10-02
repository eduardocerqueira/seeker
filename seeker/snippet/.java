//date: 2023-10-02T17:03:11Z
//url: https://api.github.com/gists/b9f4a5d49dd843965a5f9368a469a069
//owner: https://api.github.com/users/jaimemin

package com.tistory.jaimemin.springjooqwithjpa;

import com.tistory.jaimemin.jooq.code.jooq.tables.records.MemberRecord;
import com.tistory.jaimemin.jooq.code.jooq.tables.records.TeamRecord;
import com.tistory.jaimemin.springjooqwithjpa.entity.Member;
import com.tistory.jaimemin.springjooqwithjpa.entity.Team;
import com.tistory.jaimemin.springjooqwithjpa.repository.TeamJPARepository;
import org.jooq.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.tistory.jaimemin.jooq.code.jooq.tables.Member.MEMBER;
import static com.tistory.jaimemin.jooq.code.jooq.tables.Team.TEAM;

//@Transactional
@SpringBootTest
public class CRUDTest {

    private static int TEAM_CNT = 100000;

    private static int PAGE_SIZE = 1000;

    private static int MEMBER_CNT_PER_TEAM = 3;

    @Autowired
    DSLContext dslContext;

    @Autowired
    TeamJPARepository teamJPARepository;

    @Test
    void insertTeams_with_jpa() {
        List<Team> teams = new ArrayList<>();

        for (int page = 0; page < (int) Math.ceil((double) TEAM_CNT / PAGE_SIZE); page++) {
            for (int i = 0; i < PAGE_SIZE; i++) {
                Team team = new Team("Team" + (i * (page + 1) + 1)); 

                for (int j = 0; j < MEMBER_CNT_PER_TEAM; j++) {
                    Member member = Member.builder()
                            .name("Member " + (i * (page + 1) * 10 + (j + 1)))
                            .build();
                    member.setTeam(team);
                }

                teams.add(team);
            }

            teamJPARepository.saveAll(teams);
        }
    }

    @Test
    void insertTeams_with_jooq_and_jpa() {
        List<TeamRecord> teamRecords = new ArrayList<>();
        List<MemberRecord> memberRecords = new ArrayList<>();
        Map<String, String> memberName2teamName = new HashMap<>();
        List<String> teamNames = new ArrayList<>();

        for (int page = 0; page < (int) Math.ceil((double) TEAM_CNT / PAGE_SIZE); page++) {
            for (int i = 0; i < PAGE_SIZE; i++) {
                TeamRecord teamRecord = new TeamRecord();
                teamRecord.setName("Team" + (i * (page + 1) + 1));
                teamRecord.setCreatedAt(LocalDateTime.now());
                teamRecord.setModifiedAt(LocalDateTime.now());
                teamRecords.add(teamRecord);
                teamNames.add(teamRecord.getName());

                for (int j = 0; j < MEMBER_CNT_PER_TEAM; j++) {
                    MemberRecord memberRecord = new MemberRecord();
                    memberRecord.setName("Member " + (i * (page + 1) * 10 + (j + 1)));
                    memberRecord.setCreatedAt(LocalDateTime.now());
                    memberRecord.setModifiedAt(LocalDateTime.now());
                    memberRecords.add(memberRecord);
                    memberName2teamName.put(memberRecord.getName(), teamRecord.getName());
                }
            }
        }

        // Batch insert teams and get generated team IDs
        dslContext.batchInsert(teamRecords).execute();
        // Fetch generated team IDs
//        Result<TeamRecord> generatedTeams = dslContext
//                .selectFrom(TEAM)
//                .where(TEAM.NAME.in(teamNames))
//                .fetch();
        List<Team> teams = teamJPARepository.findAllByNameIn(teamNames);
        Map<String, Long> teamName2id = new HashMap<>();

//        for (TeamRecord teamRecord : generatedTeams) {
//            teamName2id.put(teamRecord.getName(), teamRecord.getId());
//        }
        for (Team team : teams) {
            teamName2id.put(team.getName(), team.getId());
        }

        for (MemberRecord memberRecord : memberRecords) {
            memberRecord.setTeamId(teamName2id.get(memberName2teamName.get(memberRecord.getName())));
        }

        // Batch insert members
        dslContext.batchInsert(memberRecords).execute();
    }

}
