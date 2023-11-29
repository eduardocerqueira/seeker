//date: 2023-11-29T17:08:59Z
//url: https://api.github.com/gists/3a3d908bfe998e20fc6c6c39e7fdd1d2
//owner: https://api.github.com/users/eod940

package com.valuewith.tweaver.groupMember.service;

import static com.valuewith.tweaver.constants.ApprovedStatus.PENDING;
import static com.valuewith.tweaver.constants.GroupStatus.OPEN;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.when;

import com.valuewith.tweaver.chat.entity.ChatRoom;
import com.valuewith.tweaver.chat.repository.ChatRoomRepository;
import com.valuewith.tweaver.constants.Provider;
import com.valuewith.tweaver.group.entity.TripGroup;
import com.valuewith.tweaver.group.repository.TripGroupRepository;
import com.valuewith.tweaver.groupMember.entity.GroupMember;
import com.valuewith.tweaver.groupMember.repository.GroupMemberRepository;
import com.valuewith.tweaver.member.entity.Member;
import com.valuewith.tweaver.member.service.MemberService;
import com.valuewith.tweaver.message.dto.MessageDto;
import com.valuewith.tweaver.message.entity.Message;
import com.valuewith.tweaver.message.repository.MessageRepository;
import com.valuewith.tweaver.message.service.MessageService;
import java.time.LocalDate;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.AdditionalAnswers;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.context.ApplicationEventPublisher;

@ExtendWith(MockitoExtension.class)
class GroupMemberApplicationServiceTest {

  @InjectMocks
  private GroupMemberApplicationService groupMemberApplicationService;
  @InjectMocks
  private MessageService messageService;

  @Mock
  private TripGroupRepository tripGroupRepository;
  @Mock
  private GroupMemberRepository groupMemberRepository;
  @Mock
  private ChatRoomRepository chatRoomRepository;
  @Mock
  private MemberService memberService;
  @Mock
  private MessageRepository messageRepository;
  @Mock
  private ApplicationEventPublisher applicationEventPublisher;

  Member alice = Member.builder()
      .memberId(1L)
      .email("alice@a.a")
      .password("a")
      .nickName("alice")
      .age(21)
      .gender("male")
      .profileUrl("http://a")
      .provider(Provider.NORMAL)
      .refreshToken("a")
      .build();

  Member bob = Member.builder()
      .memberId(1L)
      .email("bob@b.b")
      .password("b")
      .nickName("bob")
      .age(12)
      .gender("female")
      .profileUrl("http://b")
      .provider(Provider.NORMAL)
      .refreshToken("b")
      .build();

  TripGroup aliceGroup = TripGroup.builder()
      .tripGroupId(1L)
      .name("앨리스가 만든 그룹")
      .content("alice")
      .maxMemberNumber(4)
      .currentMemberNumber(1)
      .tripArea("a")
      .tripDate(LocalDate.now())
      .dueDate(LocalDate.MAX)
      .member(alice)
      .status(OPEN)
      .build();

  ChatRoom aliceChatRoom = ChatRoom.builder()
      .chatRoomId(1L)
      .tripGroup(aliceGroup)
      .title("엘리스가 만든 그룹 챗방")
      .build();

  GroupMember memberBob = GroupMember.builder()
      .groupMemberId(1L)
      .approvedStatus(PENDING)
      .member(bob)
      .tripGroup(aliceGroup)
      .build();

  @Test
  @DisplayName("성공: 그룹 참여 승인시 채팅방 참여 메시지 보내기")
  void confirm_application() {

    Message createdMessage = Message.from(aliceChatRoom, bob, "입장");

    given(messageService.createMessage(aliceChatRoom, bob, "입장"))
        .willReturn(MessageDto.from(createdMessage));

    given(messageRepository.save(createdMessage))
        .willReturn(createdMessage);

    given(memberService.findMemberByMemberId(memberBob.getGroupMemberId()))
        .willReturn(bob);

    when(tripGroupRepository.save(memberBob.getTripGroup()))
        .then(AdditionalAnswers.returnsFirstArg());

    groupMemberApplicationService.sendHelloMessage(aliceChatRoom, memberBob);

  }
}