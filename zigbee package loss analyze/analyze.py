import sys

packet_buffer = {}
packet_sum = {}

class Packet_Sum:
    def __init__(self, total_send, total_ack):
        self.total_send = total_send
        self.total_ack = total_ack
        self.str_rate = ""

class Packet:
    def __init__(self, time_seq, mac_seq):
        self.time_seq = time_seq
        self.mac_seq = mac_seq

def add_packet_buffer(key, time_seq, mac_seq):
    packet_list = packet_buffer.get(key)
    packet = Packet(time_seq, mac_seq)
    if packet_list == None:
        packet_list = []
        packet_list.append(packet)
        packet_buffer[key] = packet_list
    else:
        packet_list.append(packet)

    add_packet_send_sum(key)

def add_packet_send_sum(key):
    item = packet_sum.get(key)
    if item == None:
        item = Packet_Sum(1, 0)
        packet_sum[key] = item
    else:
        item.total_send = item.total_send + 1

def add_packet_ack_sum(key):
    item = packet_sum.get(key)
    if item != None:
        item.total_ack = item.total_ack + 1

def process_ack_frame(time_seq, mac_seq):
    keys = packet_buffer.keys()
    for key in keys:
        items = packet_buffer.get(key)
        for item in items:
            if time_seq - item.time_seq > 1:
                items.remove(item)
            else:
                if mac_seq == item.mac_seq:
                    add_packet_ack_sum(key)
                    items.remove(item)

def process_frame(line):
    line_split = line.split('\t')
    if line_split[1] == "": #abnormal frame
        return
    frame_type = int(line_split[1],16)

    if frame_type == 0x0: #not process beacon frame
        return

    if line_split[2] == "": #abnormal frame
        return

    time_seq = float(line_split[0])
    mac_seq = int(line_split[2])

    if frame_type == 0x2:
        process_ack_frame(time_seq, mac_seq)
        return

    #process other frame
    if line_split[3] == "": #abnormal frame
		return
    dst_pan_id = int(line_split[3], 16)
    if line_split[4] == "": #abnormal frame
		return
    dst_addr = int(line_split[4], 16)
    if dst_addr == 0xffff: #not process broadcast frame
        return
    try:
        src_addr = int(line_split[5], 16)
    except Exception:
        return

    str_key = hex(dst_pan_id) + ":" + hex(src_addr) + "-" + hex(dst_addr)
    add_packet_buffer(str_key, time_seq, mac_seq)

def print_summary():
    keys = packet_sum.keys()
    for key in keys:
        item = packet_sum.get(key)
        item.str_rate = "{0:.0f}%".format(float(item.total_ack)/float(item.total_send) * 100)
        print key + ":" + " Send:" + str(item.total_send) + " Ack:" + str(item.total_ack) + "--------Rate:" + item.str_rate

def process_file(in_file):
        packet_buffer.clear()
        packet_sum.clear()

	with open(in_file) as f:
		for line in f:
			if len(line) > 1: #FIXME! not know why can get last line with length 1
	#			print line
				process_frame(line)
	print_summary()
	return packet_sum

if __name__ == '__main__':
	process_file("zigbee.output")
