#date: 2023-03-29T17:42:49Z
#url: https://api.github.com/gists/a192ec33f51f90645d840bd431df6c39
#owner: https://api.github.com/users/djhohnstein

#
# rproxicmp
#
# GuidePoint Security LLC
#
# Threat and Attack Simulation Team
#
import click
import asyncio
import ipaddress
import click_params
import lib.rproxicmp_rpc
import lib.rproxicmp_arg
import lib.rproxicmp_static
import lib.rproxicmp_tunnel

from asysocks.protocol.socks5 import SOCKS5Nego
from asysocks.protocol.socks5 import SOCKS5Reply
from asysocks.protocol.socks5 import SOCKS5Method
from asysocks.protocol.socks5 import SOCKS5Request
from asysocks.protocol.socks5 import SOCKS5Command
from asysocks.protocol.socks5 import SOCKS5ReplyType
from asysocks.protocol.socks5 import SOCKS5NegoReply
from asysocks.protocol.socks5 import SOCKS5AddressType

class SockServer:
    def __init__( self, bind_addr, bind_port, timeout, rpc_host, rpc_port, agent_id ):
        """
        Initialies the internal variables the socks server uses to tunnel
        the connection over connected agent.
        """
        self.bind_addr = str( bind_addr )
        self.bind_port = bind_port
        self.timeout  = timeout
        self.rpc_host = str( rpc_host )
        self.rpc_port = rpc_port
        self.agent_id = agent_id

    async def rd_agn_wr_cli( self, rpc_obj, agent_id, remote_sock, event, writer ):
        """
        Reads from the remote file descriptor, and writes the 
        incoming data over the writer.
        """
        try:
            # have we been signalled to stop?
            while not event.is_set():
                # Read what we can from the tunnel.
                buf = await lib.rproxicmp_tunnel.tunnel_recv_async( rpc_obj, agent_id, remote_sock );

                # we have data
                if buf != b'':
                    # write the result to the client socket
                    writer.write( buf )

                    # drain the socket
                    await writer.drain()
        except Exception as e:
            print( e )
        finally:
            # set the event to stop
            event.set()

            # return
            return

    async def rd_cli_wr_agn( self, rpc_obj, agent_id, remote_sock, event, reader ):
        """
        Reads from the client and writes the result over the
        remote file descriptor.
        """
        try:
            # we have not been signalled to stop
            while not event.is_set():
                # read what we can from the buffer
                buf = await reader.read( lib.rproxicmp_static.PROXY_MAX_LENGTH )

                # we have no data
                if buf == b'' or buf is None:
                    return

                # data recieved
                await lib.rproxicmp_tunnel.tunnel_send_async( rpc_obj, agent_id, remote_sock, buf )
        except Exception as e:
            print( e )
        finally:
            # set the event to stop
            event.set()

            # return
            return

    async def handle_socks( self, reader, writer ):
        """
        Determines if the incoming request is a SOCKS5 request. If it is
        it will try to proxy the request.
        """
        try:
            tmp = await asyncio.wait_for( reader.readexactly( 1 ), timeout = self.timeout )
        except asyncio.exceptions.IncompleteReadError:
            print( 'client terminated the socket before the socks negotiation completed.' )

        # is this socks5?
        if tmp == b'\x05':
            try:
                nmt = await asyncio.wait_for( reader.readexactly( 1 ), timeout = self.timeout )
                tmt = int.from_bytes( nmt, byteorder = 'big', signed = False )
                met = await asyncio.wait_for( reader.readexactly( tmt ), timeout = self.timeout )
                cmd = SOCKS5Nego.from_bytes( tmp + nmt + met )

                # did we not recieve a no-auth command?
                if SOCKS5Method.NOAUTH not in cmd.METHODS:
                    rep = SOCKS5NegoReply.construct( SOCKS5Method.NOTACCEPTABLE );
                    writer.write( rep.to_bytes() )
                    await writer.drain()
                    return

                # send auth success
                rep = SOCKS5NegoReply.construct( SOCKS5Method.NOAUTH )
                writer.write( rep.to_bytes() )
                await writer.drain()

                # read the incoming command
                req = await asyncio.wait_for( SOCKS5Request.from_streamreader( reader ), timeout = self.timeout )

                # did we recieve a connect request?
                if req.CMD == SOCKS5Command.CONNECT and req.ATYP != SOCKS5AddressType.IP_V6:
                    # connect to the target host:port
                    rpc = lib.rproxicmp_rpc.RpcClient( self.rpc_host, self.rpc_port )

                    if req.ATYP == SOCKS5AddressType.DOMAINNAME:
                        # lookup the internal domain name
                        ip4 = await lib.rproxicmp_tunnel.tunnel_dns_lookup_async( rpc, self.agent_id, req.DST_ADDR )
                    else:
                        # use the passed address
                        ip4 = req.DST_ADDR

                    try:
                        # open a remote socket to the host
                        rfd = await lib.rproxicmp_tunnel.tunnel_connect_async( rpc, self.agent_id, ip4, req.DST_PORT )
                    except Exception:
                        # notify we could not connect
                        rep = SOCKS5Reply.construct( SOCKS5ReplyType.FAILURE, req.DST_ADDR, req.DST_PORT )
                        writer.write( rep.to_bytes() )
                        await writer.drain()
                        return
                    else:
                        # notify we connected successfully!
                        rep = SOCKS5Reply.construct( SOCKS5ReplyType.SUCCEEDED, req.DST_ADDR, req.DST_PORT )
                        writer.write( rep.to_bytes() )
                        await writer.drain()

                    # create the stop event
                    evt = asyncio.Event()

                    # create the read/write loops between client and server
                    ts1 = asyncio.create_task( self.rd_cli_wr_agn( rpc, self.agent_id, rfd, evt, reader ) )
                    ts2 = asyncio.create_task( self.rd_agn_wr_cli( rpc, self.agent_id, rfd, evt, writer ) )

                    # wait for the event to be signalled
                    await evt.wait()

                    # cancel
                    ts1.cancel()
                    ts2.cancel()

                    # close the remote socket
                    await lib.rproxicmp_tunnel.tunnel_close_async( rpc, self.agent_id, rfd );


                    # close the client socket
                    writer.close()

                    # return
                    return
            except Exception as e:
                print( e )
        else:
            print( 'client requested an unsupported protocol.' )

    async def run( self ):
        """
        Starts the SOCKS server to handle the incoming connections.
        """
        srv = await asyncio.start_server( self.handle_socks, self.bind_addr, self.bind_port )
        await srv.wait_closed()

@click.command( name = 'rproxicmp-socks', short_help = 'Creates a SOCKS5 proxy server to tunnel over an agent.' )
@lib.rproxicmp_arg.rproxicmp_arg
@click.option( '--agent-id', required = True, type = int, help = 'Agent identifier' )
@click.option( '--socks-host', type = click_params.IPV4_ADDRESS, help = 'Address to bind the SOCKS5 server to.', show_default = True, default = '127.0.0.1' )
@click.option( '--socks-port', type = int, help = 'Port to bind the SOCKS5 server to.', required = True )
@click.option( '--socks-timeout', type = int, help = 'Number of seconds to allow for a client to timeout.', required = True )
def rproxicmp_socks( rpc_host, rpc_port, agent_id, socks_host, socks_port, socks_timeout ):
    """
    Creates a SOCKS5 server that will tunnel any connected client over a
    ICMP agent connected to the RPROXICMP server. The agent will forward
    the connection from its address to the requested host:port it can 
    connect to.

    Useful for pivoting into internal networks, or combining other tools
    that may not be available through rogue.
    """
    Srv = SockServer( socks_host, socks_port, socks_timeout, rpc_host, rpc_port, agent_id )
    asyncio.run( Srv.run() )