#date: 2022-02-11T17:05:57Z
#url: https://api.github.com/gists/6208d18a55cb037aadbc6c2b00cf71e8
#owner: https://api.github.com/users/Chiggy-Playz

class PaginatedComponentsMenu(View):

    if TYPE_CHECKING:
        context: commands.Context
        children: List[Button]
        message: discord.Message

    async def on_timeout(self):
        self.quit_button.label = "Timed out"
        for child in self.children:

            child.disabled = True
        try:
            if not self.clear_on_timeout:
                await self.message.edit(view=self)
            else:
                await self.message.edit(view=None)
        except:
            pass

    async def on_error(self, *args: Any) -> None:
        await self.bot.on_error("on_interaction", *args)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user != self.author:
            await interaction.response.send_message(
                "This isn't your message. You cannot interact with it.", ephemeral=True
            )
            return False
        return True

    async def pre_hook(self, interaction: discord.Interaction):
        pass

    async def post_hook(self, interaction: discord.Interaction):
        pass

    async def beginning_cb(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.pre_hook(interaction)

        if self.next_button.disabled:
            self.next_button.disabled = False
            self.end_button.disabled = False

        self.index = 0
        self.beginning_button.disabled = True
        self.prev_button.disabled = True
        assert interaction.message is not None
        await self.post_hook(interaction)
        await interaction.followup.edit_message(
            interaction.message.id, content=self.content[self.index], embed=self.embeds[self.index], view=self
        )
        self.message = interaction.message

    async def prev_cb(self, interaction: discord.Interaction):

        await interaction.response.defer()
        await self.pre_hook(interaction)
        if self.next_button.disabled:
            self.next_button.disabled = False
            self.end_button.disabled = False

        if self.index != 0:
            self.index -= 1
            if self.index == 0:
                self.prev_button.disabled = True
                self.beginning_button.disabled = True
            assert interaction.message is not None
            await self.post_hook(interaction)
            await interaction.followup.edit_message(
                interaction.message.id, content=self.content[self.index], embed=self.embeds[self.index], view=self
            )
            self.message = interaction.message

    async def next_cb(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.pre_hook(interaction)
        if self.prev_button.disabled:
            self.prev_button.disabled = False
            self.beginning_button.disabled = False

        if self.index != (len(self.embeds) - 1):
            self.index += 1

            if self.index == (len(self.embeds) - 1):
                self.next_button.disabled = True
                self.end_button.disabled = True
            assert interaction.message is not None
            await self.post_hook(interaction)
            await interaction.followup.edit_message(
                interaction.message.id, content=self.content[self.index], embed=self.embeds[self.index], view=self
            )
            self.message = interaction.message

    async def end_cb(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self.pre_hook(interaction)
        if self.prev_button.disabled:
            self.prev_button.disabled = False
            self.beginning_button.disabled = False

        self.index = len(self.embeds) - 1
        self.end_button.disabled = True
        self.next_button.disabled = True
        assert interaction.message is not None
        await self.post_hook(interaction)
        await interaction.followup.edit_message(
            interaction.message.id, content=self.content[self.index], embed=self.embeds[self.index], view=self
        )
        self.message = interaction.message

    async def quit_cb(self, interaction: discord.Interaction):
        assert interaction.message is not None
        await interaction.message.delete()

        self.stop()

    def __init__(
        self,
        bot: HLNABot,
        author: Union[discord.User, discord.Member],
        content: Sequence[str] = tuple(),
        embeds: Sequence[Optional[discord.Embed]] = tuple(),
        clear_on_timeout: bool = False,
        allow_quit: bool = True,
        timeout: int = 60,
    ) -> None:  # sourcery skip: simplify-empty-collection-comparison
        super().__init__(timeout=timeout)

        self.bot = bot
        self.author = author
        self.index: int = 0
        self.clear_on_timeout = clear_on_timeout

        content = tuple(content)
        embeds = tuple(embeds)
        if (content == ()) and (embeds == ()):
            raise ValueError("You must provide either content or embeds.")
        if (content != ()) and (embeds == ()):
            embeds = (None,) * len(content)
        elif content == ():
            content = ("",) * len(embeds)
        elif len(content) > len(embeds):
            assert isinstance(embeds, tuple)
            embeds += (None,) * (len(content) - len(embeds))
        elif len(content) < len(embeds):
            assert isinstance(content, tuple)
            content += ("",) * (len(embeds) - len(content))

        self.embeds = embeds
        self.content = content

        self.beginning_button = Button(label="≪", style=ButtonStyle.grey, disabled=True)
        self.prev_button = Button(label="Back", style=ButtonStyle.blurple, disabled=True)
        self.next_button = Button(label="Next", style=ButtonStyle.blurple)
        self.end_button = Button(label="≫", style=ButtonStyle.grey)
        self.quit_button = Button(label="Quit", style=ButtonStyle.danger)

        self.beginning_button.callback = self.beginning_cb
        self.prev_button.callback = self.prev_cb
        self.next_button.callback = self.next_cb
        self.end_button.callback = self.end_cb
        self.quit_button.callback = self.quit_cb

        if len(embeds) > 1:
            self.add_item(self.beginning_button)
            self.add_item(self.prev_button)
            self.add_item(self.next_button)
            self.add_item(self.end_button)
            if allow_quit:
                self.add_item(self.quit_button)

    async def start(self, send: Callable, wait: bool = False):
        message = await send(content=self.content[self.index], embed=self.embeds[self.index], view=self)
        assert message
        self.message = message
        if wait:
            await self.wait()
