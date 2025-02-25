import argparse


def parse_pdhmm_file(input_file):
    reads = set()
    haplotypes = set()
    expected_results = {}
    with open(input_file, "r") as file:
        for line in file:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) == 8:
                (
                    hap_bases,
                    hap_pd_bases,
                    read_bases,
                    read_qual,
                    read_ins_qual,
                    read_del_qual,
                    gcp,
                    expected_result,
                ) = parts
                read_info = (read_bases, read_qual, read_ins_qual, read_del_qual, gcp)
                haplotype_info = (hap_bases, hap_pd_bases)
                reads.add(read_info)
                haplotypes.add(haplotype_info)
                expected_results[(read_info, haplotype_info)] = expected_result
    print("Total Reads = ", len(reads))
    print("Total Haplotypes = ", len(haplotypes))
    return reads, haplotypes, expected_results


def write_testcase_file(output_file, reads, haplotypes, expected_results):
    with open(output_file, "w") as file:
        file.write("# read-bases\tread-qual\tread-ins-qual\tread-del-qual\tgcp\n")
        for read in reads:
            file.write(f"{read[0]}\t{read[1]}\t{read[2]}\t{read[3]}\t{read[4]}\n")

        file.write("# hap-bases\thap-pd-bases\n")
        for haplotype in haplotypes:
            file.write(f"{haplotype[0]}\t{haplotype[1]}\n")

        file.write("# expected-results\n")
        for read in reads:
            for haplotype in haplotypes:
                expected_result = expected_results.get((read, haplotype), "0")
                file.write(f"{expected_result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse engineer a PDHMM file.")
    parser.add_argument("input_file", help="Path to the input PDHMM file.")
    parser.add_argument("output_file", help="Path to the output file.")
    args = parser.parse_args()

    reads, haplotypes, expected_results = parse_pdhmm_file(args.input_file)
    write_testcase_file(args.output_file, reads, haplotypes, expected_results)
