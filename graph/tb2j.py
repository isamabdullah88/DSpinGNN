from xml.etree import ElementTree as ET

class TB2JParser:
    """
    Handles the extraction of exchange parameters from TB2J XML outputs.
    Parses exchange values in meV and converts TB2J's 1-based indices to 0-based Python indices
    for easier integration with Python data structures.
    """
    @staticmethod
    def parse(xmlpath):
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        
        exchange = {}
        spinexchange = root.find('spin_exchange_list')
        
        for interaction in spinexchange.findall('spin_exchange_term'):
            ijR = interaction.find('ijR').text.split()
            # Convert 1-based TB2J indices to 0-based Python indices immediately
            i, j, Rx, Ry, Rz = int(ijR[0])-1, int(ijR[1])-1, int(ijR[2]), int(ijR[3]), int(ijR[4])
            
            # Convert to meV
            jval = float(interaction.find('data').text.split(' ')[0]) * 1000.0
            exchange[i, j, Rx, Ry, Rz] = jval
            
        return exchange