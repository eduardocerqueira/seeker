#date: 2023-08-02T16:52:27Z
#url: https://api.github.com/gists/443913589bae8a8a4cd5016e3ffb68bc
#owner: https://api.github.com/users/orestisfoufris

'''
The folowing snippet mines a new block on node 1 and then communicates that to
node 2 which is one of its peers.

It should be part of
https://github.com/bitcoin/bitcoin/blob/master/test/functional/example_test.py but I have omitted the existing
code to make the gist more readble.

If run it produces the following new log lines:

2023-08-02T16:35:44.556000Z TestFramework (INFO): node 1 top block : $11
2023-08-02T16:35:44.556000Z TestFramework (INFO): node 2 top block : $11
2023-08-02T16:35:44.559000Z TestFramework (INFO): Wait for node 2 to catch up
2023-08-02T16:35:44.563000Z TestFramework (INFO): node 1 top block : $12
2023-08-02T16:35:44.564000Z TestFramework (INFO): node 2 top block : $12
2023-08-02T16:35:44.618000Z TestFramework (INFO): Stopping nodes

'''
def create_new_block(self, tip, top_height, block_time):
  '''
  Util function to create a new block
  '''
  block = create_block(tip, create_coinbase(top_height + 1), block_time)
  block.solve()
  return block

def chaincode_challenge(self):
  # nodes 0 and 1 are already connected on setup_networks function

  node_1 = self.nodes[0]
  node_2 = self.nodes[1]

  assert node_1 != None
  assert node_2 != None

  producer = node_1.add_p2p_connection(BaseNode())

  node_1_top_block = node_1.getblockcount()
  node_2_top_block = node_1.getblockcount()

  self.log.info(f'node 1 top block : ${node_1_top_block}')
  self.log.info(f'node 2 top block : ${node_2_top_block}')

  node_1_top_block_hash = node_1.getbestblockhash()
  node_2_top_block_hash = node_1.getbestblockhash()

  # make sure nodes[0] and nodes[1] are synced
  if (node_1_top_block_hash != node_2_top_block_hash):
      self.sync_all()

  new_block = self.create_new_block(
      tip=int(node_1.getbestblockhash(), 16),
      top_height=node_1_top_block,
      block_time=node_1.getblock(node_1.getbestblockhash())['time'] + 1
  )

  new_block_message = msg_block(new_block)
  producer.send_message(new_block_message)

  self.log.info('Wait for node 2 to catch up')
  node_2.waitforblockheight(node_1_top_block + 1)

  assert_equal(node_2.getblockcount(), node_2_top_block + 1)
  assert_equal(node_2.getblockcount(), node_1.getblockcount())

  self.log.info(f'node 1 top block : ${node_1.getblockcount()}')
  self.log.info(f'node 2 top block : ${node_2.getblockcount()}')
