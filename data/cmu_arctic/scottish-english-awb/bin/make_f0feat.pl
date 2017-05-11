#!/usr/local/bin/perl
###########################################################################
##                                                                       ##
##                   Language Technologies Institute                     ##
##                     Carnegie Mellon University                        ##
##                         Copyright (c) 2003                            ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
## Antoine Raux (antoine@cs.cmu.edu) 2003                                ##
##                                                                       ##
## This script creates feature files of f0 and delta f0 values           ##
## from f0 files.                                                        ##
##                                                                       ##
###########################################################################

if (($#ARGV < 0) || ($ARGV =~ /^\-/)) {
    print STDERR "Usage: make_f0feat.pl f0/*.f0\n";
    print STDERR "\tCreates feature files of f0 and delta f0\n";
    print STDERR "\tin the f0feat directory.\n";
}


while (my $f0_fn = shift) {

    $f0_fn =~ /\b(\w+)\.f0$/ || die "$wav_fn is not a correct f0 file name!\n";
    my $base_fn = $1;
    my $f0feat_fn = "f0feat/$base_fn.f0feat";

    print STDERR "Creating $f0feat_fn...\n";

    open( TEMP, ">temp");

    my @f0_track = `ch_track $f0_fn`;

    my $f0_0 = 0.0;
    my $f0_1 = 0.0;
    my $f0_2 = 0.0;
    my $sum_f0 = 0.0;
    my @non_null_f0 = ();

    # gets the first two data points
    my $buf = &trim(shift(@f0_track));
    split( /\s+/, $buf);
    $f0_0 = $_[0];
    $buf = &trim(shift(@f0_track));
    split( /\s+/, $buf);
    $f0_1 = $_[0];

    if ($f0_0 > 0) {
	$sum_f0 += $f0_0;
	push( @non_null_f0, $f0_0);
    }
    
    # outputs the first f0 and initial delta f0
    # (value is doubled because delta is computed on
    #  a single interval instead of two)
    print TEMP "$f0_0 ".(2*($f0_1-$f0_0))."\n";
    
    while ($buf = &trim(shift(@f0_track))) {
	split( /\s+/, $buf);

	# Counts non null f0 values
	if ($f0_1 > 0) {
	    $sum_f0 += $f0_1;
	    push( @non_null_f0, $f0_1);
	}
    
	$f0_2 = $_[0];

	# outputs f0 and delta (computed between 
	# the previous and the following points)
	if (($f0_0 != 0)&&($f0_2 != 0)) {
	    print TEMP "$f0_1 ".($f0_2-$f0_0)."\n";
	}
	elsif (($f0_0 == 0)&&($f0_1 != 0)) {
	    print TEMP "$f0_1 ".(2*($f0_2-$f0_1))."\n";
	}
	elsif (($f0_2 == 0)&&($f0_1 != 0)) {
	    print TEMP "$f0_1 ".(2*($f0_1-$f0_0))."\n";
	}
	else {
	    print TEMP "$f0_1 0\n";
	}

	$f0_0 = $f0_1;
	$f0_1 = $f0_2;

	# Counts non null f0 values
	if ($f0_0 > 0) {
	    $sum_f0 += $f0_0;
	    push( @non_null_f0, $f0_0);
	}
    }
    
    # outputs the last f0 and delta f0
    # (value is doubled because delta is computed on
    #  a single interval instead of two)
    print TEMP "$f0_1 ".(2*($f0_1-$f0_0))."\n";

    close(TEMP);

    `ch_track temp -o $f0feat_fn -itype ascii -s 0.005 -otype esps`;
    `rm temp`;
}

sub trim {
    my $input = shift;

    $input =~ s/^[\s\t]+//g;
    $input =~ s/[\s\t]+$//g;

    return $input;
}

