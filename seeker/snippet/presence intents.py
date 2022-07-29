#date: 2022-07-29T17:00:59Z
#url: https://api.github.com/gists/2a64e1ff58f85973ba2156661ca8811d
#owner: https://api.github.com/users/spidirman

       
    @commands.command(aliases = ['sinfo'])
    async def serverinfo(self, ctx, guild: discord.Guild = None):
        try:
            await ctx.message.delete()
        except:
            pass

        embed = discord.Embed(title=f"{ctx.guild.name}  's information" , color=0x8FE381, timestamp=ctx.message.created_at)
        embed.set_thumbnail(url =f'{ctx.guild.icon_url}')
        embed.add_field(name = "**GENERAL INFORMATIONS :**", value = "`", inline = False)
        embed.add_field(name = "Server ID:", value = ctx.guild.id, inline = False)
        embed.add_field(name = "Owner" , value= ctx.message.guild.owner, inline = False)
        embed.add_field(name = "Region" , value= f"{ctx.guild.region}", inline = False)
        embed.add_field(name = "Verification Level" , value= f"{ctx.guild.verification_level}", inline = False)
        embed.add_field(name = "Created On", value = ctx.guild.created_at.strftime("%a, %#d %B %Y, %I:%M %p UTC"), inline = False)


        embed.add_field(name = "**CHANNEL INFORMATIONS :**", value = "`", inline = False)
        embed.add_field(name = "Total Categories" , value= f"{len(ctx.guild.categories)}", inline = False)
        embed.add_field(name = "Total Channels" , value= f"{len(ctx.guild.channels)}", inline = False)
        embed.add_field(name = "Text Channels" , value= f"{len(ctx.guild.text_channels)}", inline = False)
        embed.add_field(name = "Voice Channels" , value= f"{len(ctx.guild.voice_channels)}", inline = False)


        embed.add_field(name = "**MEMBER INFORMATIONS :**", value = "`", inline = False)
        embed.add_field(name = "Total Users" , value= f"{ctx.guild.member_count}", inline = False)
        embed.add_field(name = "online Members" , value= sum(member.status==discord.Status.online for member in ctx.guild.members), inline = False)
        embed.add_field(name = "Offline Members" , value= sum(member.status==discord.Status.offline for member in ctx.guild.members), inline = False)
        embed.add_field(name = "Humans" , value= sum(not member.bot for member in ctx.guild.members), inline = False)
        embed.add_field(name = "Bots" , value= sum(member.bot for member in ctx.guild.members), inline = False)


        embed.add_field(name = "**ROLE INFORMATIONS :**", value = "`", inline = False)
        embed.add_field(name = "Server Roles" , value= f"{len(ctx.guild.roles)}", inline = False)


        embed.add_field(name = "**BOOST INFORMATIONS :**", value = "`", inline = True)
        embed.add_field(name = "Server Boosts" , value= f"{ctx.guild.premium_subscription_count}", inline = False)
        embed.add_field(name = "Boosts Level" , value= f"{ctx.guild.premium_tier}", inline = False)
        embed.set_footer(text=f"requested by: {ctx.author}" , icon_url=ctx.author.avatar_url)
        await ctx.send(embed = embed)



    @serverinfo.error
    async def serverinfo_error(self, ctx, error):
        try:
            await ctx.message.delete()
        except:
            pass
        if isinstance(error, commands.BadArgument):
           await ctx.send(f'**{ctx.message.author.name}**, Please just use the command without arguments')
        else:
            channel = self.client.get_channel(911936164548730888)
            if discord.ChannelType.private:
                guild = ctx.author.name
            else:
                guild = ctx.guild.name
            embed = discord.Embed(title= "New Error Detected", description= f"guild: {guild}\ncommand name: {ctx.invoked_with}\nerror: {error}",color= 0xdc143c)
            await channel.send(embed=embed)
            await ctx.send(error)


    #---------------------------------------

    @commands.command()
    async def userinfo(self, ctx, *, user: discord.Member = None):
      try:
          await ctx.message.delete()
      except:
          pass
      try:
        if user is None:
            user = ctx.author
        date_format = "%a, %d %b %Y %I:%M %p"
        embed = discord.Embed(color=0xdfa3ff, description=user.mention)
        embed.set_author(name=str(user), icon_url=user.avatar_url)
        embed.set_thumbnail(url=user.avatar_url)
        embed.add_field(name="User ID:", value= user.id)
        embed.add_field(name="Joined At:", value= user.joined_at.strftime("%d/%m/%Y %H:%M:%S"))
        members = sorted(ctx.guild.members, key=lambda m: m.joined_at)
        embed.add_field(name="Join position:", value=str(members.index(user)+1))
        embed.add_field(name="Registered At:", value=user.created_at.strftime(date_format))
        if len(user.roles) > 1:
            embed.add_field(name="Roles [{}]:".format(len(user.roles)-1), value=' '.join([r.mention for r in user.roles][1:]), inline=False)
        embed.add_field(name="Guild permissions:", value=', '.join([str(p[0]).replace("", " ").title() for p in user.guild_permissions if p[1]]), inline=False)
        embed.set_footer(text=f"requested by: {ctx.author}" , icon_url=ctx.author.avatar_url)
        await ctx.send(embed = embed)
      except:
            pass

    @userinfo.error
    async def userinfo_error(self, ctx, error):
        try:
            await ctx.message.delete()
        except:
            pass
        if isinstance(error, commands.BadArgument):
           await ctx.send(f'**{ctx.message.author.name}**, Please just use a user id or user mention')
        else:
            channel = self.client.get_channel(911936164548730888)
            if discord.ChannelType.private:
                guild = ctx.author.name
            else:
                guild = ctx.guild.name
            embed = discord.Embed(title= "New Error Detected", description= f"guild: {guild}\ncommand name: {ctx.invoked_with}\nerror: {error}",color= 0xdc143c)
            await channel.send(embed=embed)
            await ctx.send(error)