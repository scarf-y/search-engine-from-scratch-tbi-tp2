import array
import math
import struct

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class RicePostings:
    """
    Rice coding (Golomb coding with M = 2^k) untuk kompresi integer list.

    Untuk menjaga agar decode selalu deterministik:
    - setiap bytestream menyimpan header berisi k dan banyak angka (n),
    - payload adalah bitstream Rice yang dipad dengan 0 ke kelipatan 8 bit.

    Header format:
    - 1 byte  : k
    - 4 bytes : n (jumlah integer yang di-encode)
    """

    HEADER_FORMAT = ">BI"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAX_K = 31

    @staticmethod
    def _choose_k(numbers):
        """Memilih parameter k sederhana berdasarkan rata-rata nilai."""
        if not numbers:
            return 0
        avg = sum(numbers) / len(numbers)
        if avg <= 1:
            return 0
        return min(max(int(math.log2(avg)), 0), RicePostings.MAX_K)

    @staticmethod
    def _rice_encode_numbers(numbers, k):
        """
        Encode list bilangan non-negatif menjadi payload Rice bit-level.
        """
        mask = (1 << k) - 1 if k > 0 else 0
        bits = []

        for number in numbers:
            if number < 0:
                raise ValueError("Rice coding hanya mendukung integer non-negatif")

            quotient = number >> k if k > 0 else number
            bits.append("1" * quotient)
            bits.append("0")
            if k > 0:
                remainder = number & mask
                bits.append(f"{remainder:0{k}b}")

        bitstream = "".join(bits)
        padding = (8 - (len(bitstream) % 8)) % 8
        if padding:
            bitstream += "0" * padding

        if not bitstream:
            return b""

        return bytes(int(bitstream[i:i + 8], 2) for i in range(0, len(bitstream), 8))

    @staticmethod
    def _rice_decode_numbers(payload, k, count):
        """
        Decode payload Rice bit-level menjadi list integer non-negatif.
        """
        if count == 0:
            return []

        bitstream = "".join(f"{byte:08b}" for byte in payload)
        numbers = []
        idx = 0

        for _ in range(count):
            quotient = 0
            while idx < len(bitstream) and bitstream[idx] == "1":
                quotient += 1
                idx += 1

            if idx >= len(bitstream):
                raise ValueError("Bitstream Rice tidak valid: terminator unary tidak ditemukan")
            idx += 1  # consume unary terminator '0'

            if k > 0:
                if idx + k > len(bitstream):
                    raise ValueError("Bitstream Rice tidak valid: sisa bit remainder kurang")
                remainder = int(bitstream[idx:idx + k], 2)
                idx += k
            else:
                remainder = 0

            numbers.append((quotient << k) + remainder)

        return numbers

    @staticmethod
    def _encode_number_list(numbers):
        k = RicePostings._choose_k(numbers)
        payload = RicePostings._rice_encode_numbers(numbers, k)
        header = struct.pack(RicePostings.HEADER_FORMAT, k, len(numbers))
        return header + payload

    @staticmethod
    def _decode_number_list(encoded_bytes):
        if len(encoded_bytes) < RicePostings.HEADER_SIZE:
            raise ValueError("Encoded bytes terlalu pendek untuk header Rice")

        k, count = struct.unpack(
            RicePostings.HEADER_FORMAT, encoded_bytes[:RicePostings.HEADER_SIZE]
        )
        payload = encoded_bytes[RicePostings.HEADER_SIZE:]
        return RicePostings._rice_decode_numbers(payload, k, count)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings list dengan transformasi gap lalu Rice coding.
        """
        if not postings_list:
            return RicePostings._encode_number_list([])

        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return RicePostings._encode_number_list(gap_postings_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings list dari bytestream Rice lalu rekonstruksi dari gap.
        """
        gap_postings_list = RicePostings._decode_number_list(encoded_postings_list)
        if not gap_postings_list:
            return []

        postings_list = [gap_postings_list[0]]
        total = gap_postings_list[0]
        for gap in gap_postings_list[1:]:
            total += gap
            postings_list.append(total)
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """Encode TF list dengan Rice coding."""
        return RicePostings._encode_number_list(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """Decode TF list dari bytestream Rice."""
        return RicePostings._decode_number_list(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, RicePostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
