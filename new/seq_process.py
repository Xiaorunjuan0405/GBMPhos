# # 原始长序列
# # long_sequence = "MASPAPPEHAEEGCPAPAAEEQAPPSPPPPQASPAERQQQEEEAQEAGAAEGAGLQVEEAAGRAAAAVTWLLGEPVLWLGCRADELLSWKRPLRSLLGFVAANLLFWFLA" \
#                 "LTPWRVYHLISVMILGRVIMQIIKDMVLSRTRGAQLWRSLSESWEVINSKPDERPRLSHCIAESWMNFSIFLQEMSLFKQQSPGKFCLLVCSVCTFFTILGSYIPGVILSY" \
#                 "LLLLCAFLCPLFKCNDIGQKIYSKIKSVLLKLDFGIGEYINQKKRERSEADKEKSHKDDSELDFSALCPKISLTVAAKELSVSDTDVSEVSWTDNGTFNLSEGYTPQTDTSDD" \
#                 "LDRPSEEVFSRDLSDFPSLENGMGTNDEDELSLGLPTELKRKKEQLDSGHRPSKETQSAAGLTLPLNSDQTFHLMSNLAGDVITAAVTAAIKDQLEGVQQALSQAAPIPEE" \
#                 "DTDTEEGDDFELLDQSELDQIESELGLTQDQEAEAQQNKKSSGFLSNLLGGH"

# long_sequence = "MNRHLWKSQLCEMVQPSGGPAADQDVLGEESPLGKPAMLHLPSEQGAPETLQRCLEENQELRDAIRQSNQILRERCEELLHFQASQREEKEFLMCKFQEARKLVERLGLEKLDLKRQKEQALREVEHLKRCQQQMAEDKASVKAQVTSLLGELQESQSRLEAATKECQALEGRARAASEQARQLESEREALQQQHSVQVDQLRMQGQSVEAALRMERQAASEEKRKLAQLQVAYHQLFQEYDNHIKSSVVGSERKRGMQLEDLKQQLQQAEEALVAKQEVIDKLKEEAEQHKIVMETVPVLKAQADIYKADFQAERQAREKLAEKKELLQEQLEQLQREYSKLKASCQESARIEDMRKRHVEVSQAPLPPAPAYLSSPLALPSQRRSPPEEPPDFCCPKCQYQAPDMDTLQIHVMECIE"
# long_sequence = "MQSTSNHLWLLSDILGQGATANVFRGRHKKTGDLFAIKVFNNISFLRPVDVQMREFEVLKKLNHKNIVKLFAIEEETTTRHKVLIMEFCPCGSLYTVLEEPSNAYGLPESEFLIVLRDVVGGMNHLRENGIVHRDIKPGNIMRVIGEDGQSVYKLTDFGAARELEDDEQFVSLYGTEEYLHPDMYERAVLRKDHQKKYGATVDLWSIGVTFYHAATGSLPFRPFEGPRRNKEVMYKIITGKPSGAISGVQKAENGPIDWSGDMPVSCSLSRGLQVLLTPVLANILEADQEKCWGFDQFFAETSDILHRMVIHVFSLQQMTAHKIYIHSYNTATIFHELVYKQTKIISSNQELIYEGRRLVLEPGRLAQHFPKTTEENPIFVVSREPLNTIGLIYEKISLPKVHPRYDLDGDASMAKAITGVVCYACRIASTLLLYQELMRKGIRWLIELIKDDYNETVHKKTEVVITLDFCIRNIEKTVKVYEKLMKINLEAAELGEISDIHTKLLRLSSSQGTIETSLQDIDSRLSPGGSLADAWAHQEGTHPKDRNVEKLQVLLNCMTEIYYQFKKDKAERRLAYNEEQIHKFDKQKLYYHATKAMTHFTDECVKKYEAFLNKSEEWIRKMLHLRKQLLSLTNQCFDIEEEVSKYQEYTNELQETLPQKMFTASSGIKHTMTPIYPSSNTLVEMTLGMKKLKEEMEGVVKELAENNHILERFGSLTMDGGLRNVDCL"
# long_sequence = "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK"
#
# # 按照规则提取短序列
# def extract_short_sequences(sequence, char='S', window_size=17):
#     short_sequences = []
#     char_indices = [i for i, c in enumerate(sequence) if c == char]
#
#     for index in char_indices:
#         # 向前提取
#         start_index_forward = max(0, index - window_size + 1)
#         short_sequence_forward = 'X' * (window_size - (index - start_index_forward + 1)) + sequence[start_index_forward:index + 1]
#
#         # 向后提取
#         end_index_backward = min(len(sequence), index + window_size)
#         short_sequence_backward = sequence[index:end_index_backward] + 'X' * (window_size - (end_index_backward - index))
#
#         # 拼接得到完整的短序列
#         short_sequence = short_sequence_forward + short_sequence_backward[1:]
#         short_sequences.append(short_sequence)
#
#     return short_sequences
#
# # 调用函数生成所有短序列
# short_sequences = extract_short_sequences(long_sequence, char='S', window_size=17)
#
# # 输出所有短序列
# # 输出所有短序列
# for i, short_sequence in enumerate(short_sequences, 1):
#     print(f">{i}|data|{short_sequence}")





# def extract_centered_sequences(sequence, center_char, window_size, padding_char='X'):
#     centered_sequences = []
#     center_indices = [i for i, c in enumerate(sequence) if c == center_char]
#
#     for index in center_indices:
#         start_index = max(0, index - window_size // 2)
#         end_index = min(len(sequence), index + window_size // 2 + 1)
#
#         centered_sequence = sequence[start_index:end_index].rjust(window_size, padding_char)
#         centered_sequences.append(centered_sequence)
#
#     return centered_sequences
#
# def generate_and_print_sequences(long_sequence):
#     # 以S为中心的短序列
#     s_centered_sequences = extract_centered_sequences(long_sequence, center_char='S', window_size=33)
#
#     # 以T为中心的短序列
#     t_centered_sequences = extract_centered_sequences(long_sequence, center_char='T', window_size=33)
#
#     # 输出以S为中心的序列
#     print("Sequences centered around 'S':")
#     for i, sequence in enumerate(s_centered_sequences, 1):
#         print(f">{i}|data|{sequence}")
#
#     # 输出以T为中心的序列
#     print("\nSequences centered around 'T':")
#     for i, sequence in enumerate(t_centered_sequences, 1):
#         print(f">{i}|data|{sequence}")

# def extract_and_print_centered_sequences(sequence, center_chars, window_size, padding_char='X'):
#     for center_char in center_chars:
#         center_indices = [i for i, c in enumerate(sequence) if c == center_char]
#
#         for index in center_indices:
#             start_index = max(0, index - window_size // 2)
#             end_index = min(len(sequence), index + window_size // 2 + 1)
#
#             centered_sequence = sequence[start_index:end_index].rjust(window_size, padding_char)
#             print(f">{center_char}|data|{centered_sequence}")
#
# # 调用方法生成并打印序列
# extract_and_print_centered_sequences(long_sequence, center_chars=['S', 'T'], window_size=33)


# # 调用方法生成并打印序列
# generate_and_print_sequences(long_sequence)




# long_sequence = "MASPAPPEHAEEGCPAPAAEEQAPPSPPPPQASPAERQQQEEEAQEAGAAEGAGLQVEEA" \
#                 "AGRAAAAVTWLLGEPVLWLGCRADELLSWKRPLRSLLGFVAANLLFWFLALTPWRVYHLI" \
#                 "SVMILGRVIMQIIKDMVLSRTRGAQLWRSLSESWEVINSKPDERPRLSHCIAESWMNFSI" \
#                 "FLQEMSLFKQQSPGKFCLLVCSVCTFFTILGSYIPGVILSYLLLLCAFLCPLFKCNDIGQ" \
#                 "KIYSKIKSVLLKLDFGIGEYINQKKRERSEADKEKSHKDDSELDFSALCPKISLTVAAKE" \
#                 "LSVSDTDVSEVSWTDNGTFNLSEGYTPQTDTSDDLDRPSEEVFSRDLSDFPSLENGMGTN" \
#                 "DEDELSLGLPTELKRKKEQLDSGHRPSKETQSAAGLTLPLNSDQTFHLMSNLAGDVITAA" \
#                 "VTAAIKDQLEGVQQALSQAAPIPEEDTDTEEGDDFELLDQSELDQIESELGLTQDQEAEA" \
#                 "QQNKKSSGFLSNLLGGH"
long_sequence = "MDKQNSQMNASHPETNLPVGYPPQYPPTAFQGPPGYSGYPGPQVSYPPPPAGHSGPGPAFPVPNQPVYNQPVYNQPVGAAGVPWMPAPQPPLNCPPGLEYLSQIDQILIHQQIELLEVLTGFETNNKYEIKNSFGQRVYFAAEDTDCCTRNCCGPSRPFTLRIIDNMGQEVITLERPLRCSSCCCPCCLQEIEIQAPPGVPIGYVIQTWHPCLPKFTIQNEKREDVLKISGPCVVCSCGDVDFEIKSLDEQCVVGKISKHWTGILREAFTDADNFGIQFPLDLDVKMKAVMIGACFLIDFMFFESTGSQEQKSGVW"

def extract_and_print_centered_sequences(sequence, center_chars, window_size, padding_char='X'):
    for center_char in center_chars:
        center_indices = [i for i, c in enumerate(sequence) if c == center_char]

        for index in center_indices:
            start_index = max(0, index - window_size // 2)
            end_index = min(len(sequence), index + window_size // 2 + 1)

            centered_sequence = sequence[start_index:end_index].rjust(window_size, padding_char)
            print(f">{center_char}|data|{centered_sequence}")

# 调用方法生成并打印序列
extract_and_print_centered_sequences(long_sequence, center_chars=['T'], window_size=33)

# sequence = "MKMASTRCKLARYLEDLEDVDLKKFKMHLEDYPPQKGCIPLPRGQTEKADHVDLATLMIDFNGEEKAWAMAVWIFAAINRRDLYEKAKRDEPKWGSDNARVSNPTVICQEDSIEEEWMGLLEYLSRISICKMKKDYRKKYRKYVRSRFQCIEDRNARLGESVSLNKRYTRLRLIKEHRSQQEREQELLAIGKTKTCESPVSPIKMELLFDPDDEHSEPVHTVVFQGAAGIGKTILARKMMLDWASGTLYQDRFDYLFYIHCREVSLVTQRSLGDLIMSCCPDPNPPIHKIVRKPSRILFLMDGFDELQGAFDEHIGPLCTDWQKAERGDILLSSLIRKKLLPEASLLITTRPVALEKLQHLLDHPRHVEILGFSEAKRKEYFFKYFSDEAQARAAFSLIQENEVLFTMCFIPLVCWIVCTGLKQQMESGKSLAQTSKTTTAVYVFFLSSLLQPRGGSQEHGLCAHLWGLCSLAADGIWNQKILFEESDLRNHGLQKADVSAFLRMNLFQKEVDCEKFYSFIHMTFQEFFAAMYYLLEEEKEGRTNVPGSRLKLPSRDVTVLLENYGKFEKGYLIFVVRFLFGLVNQERTSYLEKKLSCKISQQIRLELLKWIEVKAKAKKLQIQPSQLELFYCLYEMQEEDFVQRAMDYFPKIEINLSTRMDHMVSSFCIENCHRVESLSLGFLHNMPKEEEEEEKEGRHLDMVQCVLPSSSHAACSHGLVNSHLTSSFCRGLFSVLSTSQSLTELDLSDNSLGDPGMRVLCETLQHPGCNIRRLWLGRCGLSHECCFDISLVLSSNQKLVELDLSDNALGDFGIRLLCVGLKHLLCNLKKLWLVSCCLTSACCQDLASVLSTSHSLTRLYVGENALGDSGVAILCEKAKNPQCNLQKLGLVNSGLTSVCCSALSSVLSTNQNLTHLYLRGNTLGDKGIKLLCEGLLHPDCKLQVLELDNCNLTSHCCWDLSTLLTSSQSLRKLSLGNNDLGDLGVMMFCEVLKQQSCLLQNLGLSEMYFNYETKSALETLQEEKPELTVVFEPSW"
# s_indices = [i + 1 for i, char in enumerate(sequence) if char == 'S']
#
# # 输出结果
# for index in s_indices:
#     print(f"The index of 'S' is: {index}")
